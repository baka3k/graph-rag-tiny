import argparse
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from embedding_utils import resolve_embedding_model


MCP_NAME = "graph_rag"

# Load .env if present (for NEO4J_*/QDRANT_*/EMBEDDING_MODEL).
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    pass

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS") or os.getenv("NEO4J_PASSWORD", "password")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_RAG", "pdf_chunks")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ENTITY_TYPES = [
    "ORG",
    "PRODUCT",
    "STANDARD",
    "TECH",
    "CRYPTO",
    "SECURITY",
    "PROTOCOL",
    "VEHICLE",
    "DEVICE",
    "SERVER",
    "APP",
    "CERTIFICATE",
    "KEY",
]


def _parse_entity_types(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


ENTITY_TYPES_DEFAULT = _parse_entity_types(
    os.getenv("DEFAULT_ENTITY_TYPES") or os.getenv("ENTITY_TYPES_DEFAULT")
) or DEFAULT_ENTITY_TYPES


_qdrant_client: Optional[QdrantClient] = None
_neo4j_driver = None
_embedder: Optional[SentenceTransformer] = None


def get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            _qdrant_client = QdrantClient(url=QDRANT_URL)
        else:
            _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        model_name, local_files_only = resolve_embedding_model(None, DEFAULT_EMBEDDING_MODEL)
        _embedder = SentenceTransformer(model_name, local_files_only=local_files_only)
    return _embedder


def get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    return _neo4j_driver


def qdrant_search_entity_payload(
    query_vector: List[float],
    top_k: int,
    source_id: Optional[str],
    collection: Optional[str] = None,
) -> List[Dict[str, Any]]:
    qdrant = get_qdrant()
    collection_name = collection or QDRANT_COLLECTION
    qdrant_filter = None
    if source_id:
        qdrant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="source_id",
                    match=qmodels.MatchValue(value=source_id),
                )
            ]
        )

    if hasattr(qdrant, "search"):
        hits = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
    else:
        hits = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        ).points

    payloads = []
    for h in hits:
        if not h.payload:
            continue
        payloads.append(
            {
                "score": getattr(h, "score", None),
                "text": h.payload.get("text"),
                "source_id": h.payload.get("source_id"),
                "paragraph_id": h.payload.get("paragraph_id"),
                "entity_ids": h.payload.get("entity_ids") or [],
                "entity_mentions": h.payload.get("entity_mentions") or [],
            }
        )
    return payloads


def fetch_entities_by_ids(entity_ids: List[str]) -> List[Dict[str, Any]]:
    if not entity_ids:
        return []
    with get_neo4j().session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE e.id IN $ids
            RETURN e.id AS id, e.name AS name, e.type AS type
            """,
            ids=entity_ids,
        )
        return [dict(r) for r in result]


def fetch_relations_by_entity_ids(
    entity_ids: List[str],
    entity_types: Optional[List[str]],
    related_k: int,
) -> List[Dict[str, Any]]:
    if not entity_ids or related_k <= 0:
        return []
    entity_types = entity_types or []
    with get_neo4j().session() as session:
        result = session.run(
            """
            UNWIND $ids AS id
            MATCH (e:Entity {id: id})-[r:RELATED]-(e2:Entity)
            WHERE $types = [] OR e.type IN $types OR e2.type IN $types
            RETURN e.id AS source_id, e.name AS source, e.type AS source_type,
                   r.type AS relation,
                   e2.id AS target_id, e2.name AS target, e2.type AS target_type
            LIMIT $limit
            """,
            ids=entity_ids,
            types=entity_types,
            limit=related_k,
        )
        return [dict(r) for r in result]


def _filter_entity_ids_for_expansion(
    entity_ids: List[str],
    payloads: List[Dict[str, Any]],
    min_score_to_expand: Optional[float],
    min_entity_occurrences: Optional[int],
) -> List[str]:
    if not entity_ids:
        return []
    if min_score_to_expand is not None:
        scores = [row.get("score") for row in payloads if row.get("score") is not None]
        if scores and max(scores) < min_score_to_expand:
            return []
    if min_entity_occurrences and min_entity_occurrences > 1:
        counts: Dict[str, int] = {}
        for row in payloads:
            for entity_id in row.get("entity_ids") or []:
                counts[entity_id] = counts.get(entity_id, 0) + 1
        return [eid for eid in entity_ids if counts.get(eid, 0) >= min_entity_occurrences]
    return entity_ids


def fetch_relations_with_depth(
    entity_ids: List[str],
    entity_types: Optional[List[str]],
    related_k: int,
    depth: int,
) -> List[Dict[str, Any]]:
    if not entity_ids or related_k <= 0 or depth <= 0:
        return []
    relations: List[Dict[str, Any]] = []
    seen_relations = set()
    seen_entities = set(entity_ids)
    frontier = list(entity_ids)
    remaining = related_k

    for _ in range(depth):
        if not frontier or remaining <= 0:
            break
        step_relations = fetch_relations_by_entity_ids(
            frontier, entity_types, remaining
        )
        new_frontier = set()
        for rel in step_relations:
            rel_key = (
                rel.get("source_id"),
                rel.get("relation"),
                rel.get("target_id"),
            )
            if rel_key in seen_relations:
                continue
            seen_relations.add(rel_key)
            relations.append(rel)
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            if source_id and source_id not in seen_entities:
                new_frontier.add(source_id)
            if target_id and target_id not in seen_entities:
                new_frontier.add(target_id)

        remaining = related_k - len(relations)
        if not new_frontier:
            break
        seen_entities.update(new_frontier)
        frontier = list(new_frontier)

    return relations


def _compute_heuristic_rerank_score(
    passage: Dict[str, Any],
    entity_types: List[str],
    entity_weight: float,
    type_weight: float,
    confidence_weight: float,
    length_penalty: float,
) -> float:
    base_score = passage.get("score") or 0.0
    text = passage.get("text") or ""
    mentions = passage.get("_entity_mentions") or []
    unique_entities = set()
    type_hits = 0
    confidences: List[float] = []

    for item in mentions:
        entity_id = item.get("id") or item.get("name")
        if entity_id:
            unique_entities.add(entity_id)
        if item.get("type") in entity_types:
            type_hits += 1
        conf = item.get("confidence")
        if conf is not None:
            confidences.append(float(conf))

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    penalty = length_penalty * len(text) if length_penalty else 0.0
    return (
        base_score
        + entity_weight * len(unique_entities)
        + type_weight * type_hits
        + confidence_weight * avg_conf
        - penalty
    )


def _apply_heuristic_rerank(
    passages: List[Dict[str, Any]],
    entity_types: List[str],
    entity_weight: float,
    type_weight: float,
    confidence_weight: float,
    length_penalty: float,
) -> List[Dict[str, Any]]:
    for passage in passages:
        passage["rerank_score"] = _compute_heuristic_rerank_score(
            passage,
            entity_types,
            entity_weight,
            type_weight,
            confidence_weight,
            length_penalty,
        )
    return sorted(passages, key=lambda row: row.get("rerank_score", 0.0), reverse=True)


def fetch_paragraph_by_source(
    source_id: str,
    paragraph_id: int,
) -> Optional[Dict[str, Any]]:
    with get_neo4j().session() as session:
        result = session.run(
            """
            MATCH (p:Paragraph {source_id: $source_id, paragraph_id: $paragraph_id})
            RETURN p.text AS text,
                   p.short AS short,
                   p.source_id AS source_id,
                   p.paragraph_id AS paragraph_id
            """,
            source_id=source_id,
            paragraph_id=paragraph_id,
        )
        record = result.single()
        return dict(record) if record else None




def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def list_source_ids(limit: int = 50) -> List[str]:
        """List available source_id values from Neo4j (Paragraph nodes)."""
        with get_neo4j().session() as session:
            result = session.run(
                """
                MATCH (p:Paragraph)
                WHERE p.source_id IS NOT NULL
                RETURN DISTINCT p.source_id AS source_id
                ORDER BY source_id
                LIMIT $limit
                """,
                limit=limit,
            )
            return [r["source_id"] for r in result]

    @mcp.tool()
    def list_qdrant_collections() -> List[str]:
        """List Qdrant collections."""
        qdrant = get_qdrant()
        return [c.name for c in qdrant.get_collections().collections]

    @mcp.tool()
    def query_graph_rag_langextract(
        query: str,
        top_k: int = 5,
        source_id: Optional[str] = None,
        collection: Optional[str] = None,
        include_entities: bool = True,
        include_relations: bool = True,
        expand_related: bool = True,
        related_k: int = 50,
        graph_depth: int = 1,
        entity_types: Optional[List[str]] = None,
        max_passage_chars: Optional[int] = None,
        min_score_to_expand: Optional[float] = None,
        min_entity_occurrences: Optional[int] = None,
        rerank: bool = False,
        rerank_entity_weight: float = 0.05,
        rerank_type_weight: float = 0.1,
        rerank_confidence_weight: float = 0.3,
        rerank_length_penalty: float = 0.0002,
    ) -> Dict[str, Any]:
        """
        Query Qdrant for top-k passages with entity_ids payload, then fetch related
        entity context from Neo4j. Returns context only (no LLM generation).
        """
        embedder = get_embedder()
        q_vec = embedder.encode([query])[0].tolist()

        if entity_types is None:
            entity_types = list(ENTITY_TYPES_DEFAULT)

        payloads = qdrant_search_entity_payload(
            q_vec, top_k=top_k, source_id=source_id, collection=collection
        )

        passages = []
        entity_ids: List[str] = []
        for row in payloads:
            text = row.get("text") or ""
            if max_passage_chars:
                text = text[:max_passage_chars]
            passages.append(
                {
                    "text": text,
                    "score": row.get("score"),
                    "source_id": row.get("source_id"),
                    "paragraph_id": row.get("paragraph_id"),
                    "_entity_mentions": row.get("entity_mentions") or [],
                }
            )
            entity_ids.extend(row.get("entity_ids") or [])

        entity_ids = list(dict.fromkeys(entity_ids))
        entity_ids = _filter_entity_ids_for_expansion(
            entity_ids,
            payloads,
            min_score_to_expand=min_score_to_expand,
            min_entity_occurrences=min_entity_occurrences,
        )

        entities = fetch_entities_by_ids(entity_ids) if include_entities else []
        relations = []
        if include_relations and expand_related:
            relations = fetch_relations_with_depth(
                entity_ids, entity_types, related_k, graph_depth
            )

        if rerank:
            passages = _apply_heuristic_rerank(
                passages,
                entity_types,
                rerank_entity_weight,
                rerank_type_weight,
                rerank_confidence_weight,
                rerank_length_penalty,
            )
            for passage in passages:
                passage.pop("_entity_mentions", None)
        else:
            for passage in passages:
                passage.pop("_entity_mentions", None)

        return {
            "query": query,
            "top_k": top_k,
            "source_id": source_id,
            "collection": collection or QDRANT_COLLECTION,
            "graph_depth": graph_depth,
            "min_score_to_expand": min_score_to_expand,
            "min_entity_occurrences": min_entity_occurrences,
            "rerank_applied": rerank,
            "rerank_strategy": "heuristic" if rerank else None,
            "passages": passages,
            "entities": entities,
            "relations": relations,
        }

    @mcp.tool()
    def get_paragraph_text(
        source_id: str,
        paragraph_id: int,
    ) -> Dict[str, Any]:
        """Fetch a paragraph's text by source_id + paragraph_id from Neo4j."""
        if not source_id:
            return {"warning": "source_id is required."}
        record = fetch_paragraph_by_source(source_id, paragraph_id)
        if not record:
            return {
                "source_id": source_id,
                "paragraph_id": paragraph_id,
                "text": None,
                "warning": "Paragraph not found.",
            }
        return record


if __name__ == "__main__":
    force_quit = {"armed": False}

    def _handle_sigint(signum, _frame) -> None:
        if force_quit["armed"]:
            print("Force quitting now.")
            os._exit(0)
        force_quit["armed"] = True
        if signum == signal.SIGTERM:
            print("Received SIGTERM. Send again to force quit.")
        else:
            print("Received SIGINT. Press Ctrl+C again to force quit.")

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _handle_sigint)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
    )
    parser.add_argument("--host", default=os.getenv("FASTMCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FASTMCP_PORT", "8789")))
    parser.add_argument(
        "--path",
        dest="stream_path",
        default=os.getenv("FASTMCP_STREAMABLE_HTTP_PATH", "/mcp"),
        help="Streamable HTTP path",
    )
    parser.add_argument(
        "--stream-path",
        dest="stream_path",
        default=os.getenv("FASTMCP_STREAMABLE_HTTP_PATH", "/mcp"),
        help="Streamable HTTP path (deprecated, use --path)",
    )
    args = parser.parse_args()

    mcp = FastMCP(
        MCP_NAME,
        host=args.host,
        port=args.port,
        streamable_http_path=args.stream_path,
    )
    register_tools(mcp)
    transport = args.transport
    endpoint = f"http://{args.host}:{args.port}{args.stream_path}"
    print(f"Starting MCP server: {MCP_NAME}")
    print(f"Transport: {transport}")
    if transport == "streamable-http":
        print(f"Endpoint: {endpoint}")
    else:
        print("Endpoint: (stdio)")
    mcp.run(transport=args.transport)
