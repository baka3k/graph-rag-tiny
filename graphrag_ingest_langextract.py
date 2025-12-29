import argparse
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from entity_extractors import (
    build_spacy_pipeline,
    extract_entities_gemini,
    extract_entities_langextract,
)


def _load_env() -> None:
    if load_dotenv is None:
        return
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()


def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(p for p in parts if p.strip())


def read_text_file(text_path: Path) -> str:
    return text_path.read_text(encoding="utf-8").strip()


def read_docx_text(docx_path: Path) -> str:
    doc = Document(str(docx_path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(parts)


def read_pptx_text(pptx_path: Path) -> str:
    presentation = Presentation(str(pptx_path))
    parts = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            text = getattr(shape, "text", None)
            if text:
                stripped = text.strip()
                if stripped:
                    parts.append(stripped)
    return "\n\n".join(parts)


def split_paragraphs(text: str, max_chars: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            for i in range(0, len(para), max_chars):
                chunk = para[i : i + max_chars].strip()
                if chunk:
                    chunks.append(chunk)
    return chunks


def _set_langextract_overrides(model_id: str | None, model_url: str | None) -> None:
    if model_id:
        os.environ["LANGEXTRACT_MODEL_ID"] = model_id
    if model_url:
        os.environ["LANGEXTRACT_MODEL_URL"] = model_url


def build_graph_components(
    text: str,
    provider: str,
    nlp=None,
) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]]]:
    if provider == "spacy":
        doc = nlp(text)
        entities = [{"name": ent.text, "type": ent.label_} for ent in doc.ents]
        relations = []
    elif provider == "gemini":
        entities, relations = extract_entities_gemini(text)
    else:
        entities, relations = extract_entities_langextract(text)
    nodes: Dict[str, Dict[str, str]] = {}
    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue
        nodes.setdefault(
            name,
            {
                "id": str(uuid.uuid4()),
                "name": name,
                "type": ent.get("type", "UNKNOWN") or "UNKNOWN",
            },
        )

    cleaned_relations = []
    for rel in relations:
        source = rel.get("source", "").strip()
        target = rel.get("target", "").strip()
        rel_type = rel.get("relation", "").strip() or "RELATED"
        if not source or not target:
            continue
        if source not in nodes:
            nodes[source] = {"id": str(uuid.uuid4()), "name": source, "type": "UNKNOWN"}
        if target not in nodes:
            nodes[target] = {"id": str(uuid.uuid4()), "name": target, "type": "UNKNOWN"}
        cleaned_relations.append(
            {
                "source_id": nodes[source]["id"],
                "target_id": nodes[target]["id"],
                "type": rel_type,
            }
        )
    return nodes, cleaned_relations


def ingest_to_neo4j(
    driver,
    nodes: Dict[str, Dict[str, str]],
    relations: List[Dict[str, str]],
    source_id: str,
    paragraph_id: int,
    paragraph_text: str | None = None,
    is_short: bool = False,
) -> None:
    with driver.session() as session:
        if paragraph_text:
            session.run(
                """
                MERGE (p:Paragraph {source_id: $source_id, paragraph_id: $paragraph_id})
                SET p.text = $text,
                    p.short = $is_short
                """,
                source_id=source_id,
                paragraph_id=paragraph_id,
                text=paragraph_text,
                is_short=is_short,
            )
        for node in nodes.values():
            session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.name = $name,
                    e.type = $type,
                    e.source_id = $source_id,
                    e.paragraph_id = $paragraph_id
                """,
                id=node["id"],
                name=node["name"],
                type=node["type"],
                source_id=source_id,
                paragraph_id=paragraph_id,
            )
            if paragraph_text:
                session.run(
                    """
                    MATCH (p:Paragraph {source_id: $source_id, paragraph_id: $paragraph_id})
                    MATCH (e:Entity {id: $id})
                    MERGE (p)-[:HAS_ENTITY]->(e)
                    """,
                    source_id=source_id,
                    paragraph_id=paragraph_id,
                    id=node["id"],
                )
        for rel in relations:
            session.run(
                """
                MATCH (s:Entity {id: $source_id})
                MATCH (t:Entity {id: $target_id})
                MERGE (s)-[r:RELATED {type: $type}]->(t)
                SET r.source_id = $source_doc,
                    r.paragraph_id = $paragraph_id
                """,
                source_id=rel["source_id"],
                target_id=rel["target_id"],
                type=rel["type"],
                source_doc=source_id,
                paragraph_id=paragraph_id,
            )


def create_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    try:
        client.get_collection(name)
        return
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )


def ingest_to_qdrant(
    client: QdrantClient,
    collection: str,
    paragraph: str,
    paragraph_id: int,
    embedder: SentenceTransformer,
    nodes: Dict[str, Dict[str, str]],
    source_id: str,
) -> None:
    node_lookup = {name.lower(): node["id"] for name, node in nodes.items()}
    vector = embedder.encode([paragraph])[0]
    entity_ids = []
    lower_para = paragraph.lower()
    for name_lower, node_id in node_lookup.items():
        if name_lower and name_lower in lower_para:
            entity_ids.append(node_id)
    point = qmodels.PointStruct(
        id=str(uuid.uuid4()),
        vector=vector.tolist(),
        payload={
            "paragraph_id": paragraph_id,
            "source_id": source_id,
            "text": paragraph,
            "entity_ids": entity_ids,
        },
    )
    client.upsert(collection_name=collection, points=[point])


def _safe_source_id(base: Path, file_path: Path) -> str:
    rel = file_path.relative_to(base).as_posix()
    return rel.replace("/", "__").replace("\\", "__")


def _iter_input_files(folder: Path) -> List[Path]:
    files = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md", ".docx", ".pptx"}:
            files.append(path)
    return sorted(files)


def _read_input_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_text(file_path)
    if suffix == ".docx":
        return read_docx_text(file_path)
    if suffix == ".pptx":
        return read_pptx_text(file_path)
    return read_text_file(file_path)


def process_text(
    raw_text: str,
    source_id: str,
    args: argparse.Namespace,
    driver,
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    nlp=None,
) -> None:
    if not raw_text:
        print(f"Skip empty input: {source_id}")
        return

    ingested_at = datetime.now(timezone.utc).isoformat()
    print(f"Source: {source_id} at {ingested_at}")
    print(f"Entity provider: {args.entity_provider}")

    if args.entity_provider == "spacy" and nlp is None:
        nlp = build_spacy_pipeline(args.spacy_model, ruler_json=args.ruler_json)

    paragraphs = split_paragraphs(raw_text, args.max_paragraph_chars)
    print(f"Chunked into {len(paragraphs)} paragraphs.")
    for idx, paragraph in enumerate(paragraphs):
        if not paragraph:
            continue
        if len(paragraph) < args.min_paragraph_chars:
            if args.skip_llm_short:
                print(
                    f"Paragraph {idx + 1}/{len(paragraphs)}: LLM skipped (len={len(paragraph)})"
                )
                print("Ingesting short paragraph to Neo4j...")
                ingest_to_neo4j(
                    driver,
                    {},
                    [],
                    source_id,
                    paragraph_id=idx,
                    paragraph_text=paragraph,
                    is_short=True,
                )
                print("Neo4j short paragraph stored.")
                print("Ingesting to Qdrant...")
                ingest_to_qdrant(qdrant, args.collection, paragraph, idx, embedder, {}, source_id)
                print("Qdrant ingestion complete.")
            else:
                print(f"Paragraph {idx + 1}/{len(paragraphs)}: skipped (len={len(paragraph)})")
            continue
        print(f"Paragraph {idx + 1}/{len(paragraphs)}: extracting entities/relations...")
        nodes, relations = build_graph_components(paragraph, args.entity_provider, nlp=nlp)
        print(f"Extracted {len(nodes)} entities, {len(relations)} relations.")

        print("Ingesting to Neo4j...")
        ingest_to_neo4j(
            driver,
            nodes,
            relations,
            source_id,
            paragraph_id=idx,
            paragraph_text=paragraph,
        )
        print("Neo4j ingestion complete.")

        print("Ingesting to Qdrant...")
        ingest_to_qdrant(qdrant, args.collection, paragraph, idx, embedder, nodes, source_id)
        print("Qdrant ingestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path to PDF")
    parser.add_argument("--text-file", help="Path to UTF-8 text file")
    parser.add_argument("--raw-text", help="Raw text input")
    parser.add_argument("--folder", help="Folder to scan for .pdf/.txt files (recursive)")
    parser.add_argument("--source-id", default=None, help="Custom source identifier")
    parser.add_argument("--collection", default="graphrag_entities")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-paragraph-chars", type=int, default=1200)
    parser.add_argument("--min-paragraph-chars", type=int, default=150)
    parser.add_argument(
        "--skip-llm-short",
        action="store_true",
        help="Skip LLM extraction for short paragraphs but still store in Qdrant",
    )
    parser.add_argument(
        "--entity-provider",
        choices=["spacy", "gemini", "langextract"],
        default="langextract",
        help="Entity extraction provider",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--ruler-json", default=None, help="Path to EntityRuler JSON patterns")
    parser.add_argument("--langextract-model-id", default=os.getenv("LANGEXTRACT_MODEL_ID"))
    parser.add_argument("--langextract-model-url", default=os.getenv("LANGEXTRACT_MODEL_URL"))
    parser.add_argument("--llm-debug", action="store_true", help="Print raw LLM output")
    parser.add_argument("--llm-retry-count", type=int, default=None)
    parser.add_argument("--llm-retry-backoff", type=float, default=None)
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-pass", default=os.getenv("NEO4J_PASS", "password"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_KEY"))
    args = parser.parse_args()

    _load_env()

    inputs = [bool(args.pdf), bool(args.text_file), bool(args.raw_text), bool(args.folder)]
    if sum(inputs) != 1:
        raise SystemExit("Provide exactly one of --pdf, --text-file, --raw-text, or --folder.")

    if args.llm_debug:
        os.environ["LLM_DEBUG"] = "1"
        os.environ.setdefault("LLM_DEBUG_LOG_PATH", "logs/langextract_raw.log")
    if args.llm_retry_count is not None:
        os.environ["LLM_RETRY_COUNT"] = str(args.llm_retry_count)
    if args.llm_retry_backoff is not None:
        os.environ["LLM_RETRY_BACKOFF_SECONDS"] = str(args.llm_retry_backoff)
    _set_langextract_overrides(args.langextract_model_id, args.langextract_model_url)
    embedder = SentenceTransformer(args.embedding_model)

    if args.qdrant_url:
        qdrant = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    else:
        qdrant = QdrantClient(
            host=args.qdrant_host, port=args.qdrant_port, api_key=args.qdrant_api_key
        )
    create_collection(qdrant, args.collection, vector_size=embedder.get_sentence_embedding_dimension())

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))

    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            raise FileNotFoundError(folder_path)
        file_paths = _iter_input_files(folder_path)
        if not file_paths:
            raise SystemExit("No supported files found in folder.")
        print(f"Found {len(file_paths)} files in folder.")
        shared_nlp = None
        if args.entity_provider == "spacy":
            shared_nlp = build_spacy_pipeline(args.spacy_model, ruler_json=args.ruler_json)
        for idx, file_path in enumerate(file_paths, start=1):
            print(f"[{idx}/{len(file_paths)}] Processing: {file_path}")
            if args.source_id:
                source_id = f"{args.source_id}__{_safe_source_id(folder_path, file_path)}"
            else:
                source_id = _safe_source_id(folder_path, file_path)
            raw_text = _read_input_text(file_path)
            process_text(raw_text, source_id, args, driver, qdrant, embedder, nlp=shared_nlp)
        return

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        raw_text = read_pdf_text(pdf_path)
        source_id = args.source_id or pdf_path.stem
        process_text(raw_text, source_id, args, driver, qdrant, embedder)
        return

    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            raise FileNotFoundError(text_path)
        raw_text = read_text_file(text_path)
        source_id = args.source_id or text_path.stem
        process_text(raw_text, source_id, args, driver, qdrant, embedder)
        return

    raw_text = args.raw_text.strip()
    source_id = args.source_id or "raw_text"
    process_text(raw_text, source_id, args, driver, qdrant, embedder)


if __name__ == "__main__":
    main()
