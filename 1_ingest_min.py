import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase

from embedding_utils import resolve_embedding_model

def read_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        pages.append((idx, page.extract_text() or ""))
    return pages


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\\s+", text.strip())
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
    paragraphs = [p.strip() for p in re.split(r"\\n\\s*\\n+", text) if p.strip()]
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
    parser.add_argument("--chunking", choices=["char", "sentence", "paragraph"], default="char")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection", default="pdf_chunks")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc_id = args.doc_id or pdf_path.stem
    aliases = [doc_id, pdf_path.stem, pdf_path.name, str(pdf_path)]
    aliases = [alias for alias in dict.fromkeys(aliases) if alias]
    ingested_at = datetime.now(timezone.utc).isoformat()

    pages = read_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, args.chunk_size, args.chunking)

    model_name, local_files_only = resolve_embedding_model(
        args.embedding_model, "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    vectors = model.encode([c["text"] for c in chunks])

    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    collections = [c.name for c in qdrant.get_collections().collections]
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

    print("Done: inserted chunks into Qdrant and Neo4j.")


if __name__ == "__main__":
    main()
