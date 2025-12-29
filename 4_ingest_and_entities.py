import argparse
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from entity_extractors import (
    build_spacy_pipeline,
    extract_entities_azure,
    extract_entities_gemini,
    extract_entities_langextract,
)


def read_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        pages.append((idx, page.extract_text() or ""))
    return pages


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_by_sentences(text: str, chunk_size: int) -> list[str]:
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_by_paragraph(text: str, chunk_size: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            chunks.extend(chunk_by_sentences(para, chunk_size))
    return chunks


def chunk_text(text: str, chunk_size: int, mode: str) -> list[str]:
    if mode == "sentence":
        return chunk_by_sentences(text, chunk_size)
    if mode == "paragraph":
        return chunk_by_paragraph(text, chunk_size)
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int, mode: str) -> list[dict]:
    chunks = []
    for page_num, text in pages:
        for chunk_text_value in chunk_text(text, chunk_size, mode):
            if chunk_text_value.strip():
                chunks.append({"page": page_num, "text": chunk_text_value})
    return chunks


def make_point_id(doc_id: str, chunk_id: int) -> int:
    digest = hashlib.blake2b(f"{doc_id}:{chunk_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--doc-id", default=None, help="Document id")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunking", choices=["char", "sentence", "paragraph"], default="paragraph")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--collection", default="pdf_chunks")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--ruler-json", default=None, help="Path to EntityRuler JSON patterns")
    parser.add_argument(
        "--entity-provider",
        choices=["spacy", "azure", "gemini", "langextract"],
        default="spacy",
        help="Entity extraction provider",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc_id = args.doc_id or pdf_path.stem
    aliases = [doc_id, pdf_path.stem, pdf_path.name, str(pdf_path)]
    aliases = [alias for alias in dict.fromkeys(aliases) if alias]
    ingested_at = datetime.now(timezone.utc).isoformat()

    pages = read_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, args.chunk_size, args.chunking)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode([c["text"] for c in chunks])

    if args.qdrant_url:
        qdrant = QdrantClient(url=args.qdrant_url)
        qdrant_label = args.qdrant_url
    else:
        qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
        qdrant_label = f"{args.qdrant_host}:{args.qdrant_port}"
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
    except Exception as exc:
        raise SystemExit(
            "Qdrant is not reachable. Start Qdrant or set QDRANT_URL / --qdrant-url. "
            f"Target: {qdrant_label}"
        ) from exc
    if args.collection not in collections:
        qdrant.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )

    points = []
    for i, vec in enumerate(vectors):
        point_id = make_point_id(doc_id, i)
        points.append(
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={"chunk_id": i, "doc_id": doc_id, "page": chunks[i]["page"]},
            )
        )
    qdrant.upsert(collection_name=args.collection, points=points)
    print(f"Stored {len(points)} vectors in Qdrant collection '{args.collection}'.")

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    with driver.session() as session:
        session.run(
            """
            MERGE (d:Document {id: $id})
            SET d.name = $name,
                d.aliases = $aliases,
                d.source = $source,
                d.ingested_at = $ingested_at
            """,
            id=doc_id,
            name=pdf_path.name,
            aliases=aliases,
            source=str(pdf_path),
            ingested_at=ingested_at,
        )
        for i, chunk in enumerate(chunks):
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                MERGE (c:Chunk {id: $chunk_id, doc_id: $doc_id})
                SET c.text = $text, c.page = $page
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                doc_id=doc_id,
                chunk_id=i,
                text=chunk["text"],
                page=chunk["page"],
            )
            if (i + 1) % 100 == 0 or (i + 1) == len(chunks):
                print(f"Neo4j chunks: {i + 1}/{len(chunks)} stored")

    if args.entity_provider == "spacy" and args.ruler_json:
        print(f"Extracting entities using provider: spacy + EntityRuler ({args.ruler_json})")
    else:
        print(f"Extracting entities using provider: {args.entity_provider}")
    with driver.session() as session:
        chunk_rows = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN c.id AS id, c.text AS text",
            doc_id=doc_id,
        )
        if args.entity_provider == "spacy":
            nlp = build_spacy_pipeline(args.spacy_model, ruler_json=args.ruler_json)
        for record in chunk_rows:
            chunk_id = record["id"]
            chunk_text_value = record["text"] or ""
            print(f"Entity extraction for chunk {chunk_id} (len={len(chunk_text_value)})")
            if args.entity_provider == "spacy":
                doc = nlp(chunk_text_value)
                entities = [{"name": ent.text, "type": ent.label_} for ent in doc.ents]
                relations = []
            elif args.entity_provider == "azure":
                entities, relations = extract_entities_azure(chunk_text_value)
            elif args.entity_provider == "gemini":
                entities, relations = extract_entities_gemini(chunk_text_value)
            else:
                entities, relations = extract_entities_langextract(chunk_text_value)

            for ent in entities:
                session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id, doc_id: $doc_id})
                    MERGE (e:Entity {name: $name, type: $type})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    name=ent["name"],
                    type=ent["type"],
                )

            for rel in relations:
                session.run(
                    """
                    MERGE (s:Entity {name: $source})
                    MERGE (t:Entity {name: $target})
                    MERGE (s)-[r:RELATED {type: $relation}]->(t)
                    """,
                    source=rel["source"],
                    target=rel["target"],
                    relation=rel["relation"],
                )

    print("Done: ingested PDF, stored chunks, and created entities/relations.")


if __name__ == "__main__":
    main()
