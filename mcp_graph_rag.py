import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer


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
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


_qdrant_client: Optional[QdrantClient] = None
_neo4j_driver = None
_embedder: Optional[SentenceTransformer] = None
_entity_schema_cached: Optional[Tuple[bool, bool]] = None


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
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    return _neo4j_driver


def entity_schema_available() -> Tuple[bool, bool]:
    global _entity_schema_cached
    if _entity_schema_cached is not None:
        return _entity_schema_cached

    with get_neo4j().session() as session:
        labels = {r["label"] for r in session.run("CALL db.labels() YIELD label")}
        rels = {
            r["relationshipType"]
            for r in session.run("CALL db.relationshipTypes() YIELD relationshipType")
        }
    _entity_schema_cached = ("Entity" in labels, "MENTIONS" in rels)
    return _entity_schema_cached


def qdrant_search(
    query_vector: List[float],
    top_k: int,
    doc_id: Optional[str],
    source_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    qdrant = get_qdrant()
    qdrant_filter = None
    must = []
    if doc_id:
        must.append(
            qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=doc_id),
            )
        )
    if source_id:
        must.append(
            qmodels.FieldCondition(
                key="source_id",
                match=qmodels.MatchValue(value=source_id),
            )
        )
    if must:
        qdrant_filter = qmodels.Filter(must=must)

    if hasattr(qdrant, "search"):
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
    else:
        hits = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        ).points

    payloads = []
    for h in hits:
        if not h.payload or h.payload.get("chunk_id") is None:
            continue
        payloads.append(
            {
                "doc_id": h.payload.get("doc_id"),
                "chunk_id": h.payload.get("chunk_id"),
                "score": getattr(h, "score", None),
            }
        )
    return payloads


def qdrant_search_entity_payload(
    query_vector: List[float],
    top_k: int,
    source_id: Optional[str],
) -> List[Dict[str, Any]]:
    qdrant = get_qdrant()
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
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
    else:
        hits = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
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
            }
        )
    return payloads


def fetch_chunks_by_pairs(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not pairs:
        return []
    with get_neo4j().session() as session:
        result = session.run(
            """
            UNWIND $pairs AS pair
            MATCH (c:Chunk {doc_id: pair.doc_id, id: pair.chunk_id})
            RETURN c.id AS id, c.doc_id AS doc_id, c.text AS text, c.page AS page
            """,
            pairs=pairs,
        )
        return [dict(r) for r in result]


def fetch_entities_for_pairs(
    pairs: List[Dict[str, Any]], entity_types: Optional[List[str]]
) -> List[Dict[str, Any]]:
    if not pairs:
        return []
    has_entity, has_mentions = entity_schema_available()
    if not (has_entity and has_mentions):
        return []
    entity_types = entity_types or []
    with get_neo4j().session() as session:
        result = session.run(
            """
            UNWIND $pairs AS pair
            MATCH (c:Chunk {doc_id: pair.doc_id, id: pair.chunk_id})-[:MENTIONS]->(e:Entity)
            WHERE $types = [] OR e.type IN $types
            RETURN c.id AS id, c.doc_id AS doc_id, collect(DISTINCT e.name) AS entities
            """,
            pairs=pairs,
            types=entity_types,
        )
        return [dict(r) for r in result]


def fetch_related_chunks(
    entities: List[str],
    exclude_pairs: List[Dict[str, Any]],
    related_k: int,
    related_doc_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if not entities or related_k <= 0:
        return []
    has_entity, has_mentions = entity_schema_available()
    if not (has_entity and has_mentions):
        return []

    exclude_pairs = exclude_pairs or []
    related_doc_ids = related_doc_ids or []
    with get_neo4j().session() as session:
        result = session.run(
            """
            UNWIND $exclude AS ex
            WITH collect({doc_id: ex.doc_id, chunk_id: ex.chunk_id}) AS excluded
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $entities
            AND NOT {doc_id: c.doc_id, chunk_id: c.id} IN excluded
            AND ($doc_ids = [] OR c.doc_id IN $doc_ids)
            RETURN c.id AS id, c.doc_id AS doc_id, c.text AS text, c.page AS page,
                   collect(DISTINCT e.name) AS entities
            LIMIT $limit
            """,
            entities=entities,
            exclude=exclude_pairs,
            doc_ids=related_doc_ids,
            limit=related_k,
        )
    return [dict(r) for r in result]


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


def fetch_relations_for_pairs(
    pairs: List[Dict[str, Any]], entity_types: Optional[List[str]]
) -> List[Dict[str, Any]]:
    if not pairs:
        return []
    has_entity, _has_mentions = entity_schema_available()
    if not has_entity:
        return []
    entity_types = entity_types or []
    with get_neo4j().session() as session:
        result = session.run(
            """
            UNWIND $pairs AS pair
            MATCH (c:Chunk {doc_id: pair.doc_id, id: pair.chunk_id})-[:MENTIONS]->(e:Entity)
            WHERE $types = [] OR e.type IN $types
            MATCH (e)-[r:RELATED]->(e2:Entity)
            RETURN c.id AS id, c.doc_id AS doc_id,
                   collect(DISTINCT {
                     source: e.name, source_type: e.type,
                     relation: r.type, target: e2.name, target_type: e2.type
                   }) AS relations
            """,
            pairs=pairs,
            types=entity_types,
        )
        return [dict(r) for r in result]


def resolve_doc_id(doc_ref: str) -> Optional[str]:
    """Resolve a doc reference (id, alias, or name) to a canonical doc_id."""
    if not doc_ref:
        return None
    with get_neo4j().session() as session:
        result = session.run(
            """
            MATCH (d:Document)
            WHERE d.id = $ref OR d.name = $ref OR $ref IN coalesce(d.aliases, [])
            RETURN d.id AS id
            LIMIT 1
            """,
            ref=doc_ref,
        )
        record = result.single()
        return record["id"] if record else None


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def query_graph_rag(
        query: str,
        top_k: int = 3,
        doc_id: Optional[str] = None,
        doc_ref: Optional[str] = None,
        include_entities: bool = True,
        include_relations: bool = True,
        expand_related: bool = True,
        related_scope: str = "same-doc",
        related_k: int = 3,
        entity_types: Optional[List[str]] = None,
        max_chunk_chars: Optional[int] = None,
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query Qdrant for top-k chunks, then fetch chunk text and optional entities from Neo4j.
        If expand_related is True, also fetch related chunks via shared entities.
        """
        embedder = get_embedder()
        q_vec = embedder.encode([query])[0].tolist()

        if not doc_id and doc_ref:
            doc_id = resolve_doc_id(doc_ref)

        payloads = qdrant_search(
            q_vec, top_k=top_k, doc_id=doc_id, source_id=source_id
        )
        if source_id and not payloads:
            return {
                "top_k": top_k,
                "doc_id": doc_id,
                "chunks": [],
                "entities": [],
                "relations": [],
                "related": [],
                "warning": "No chunk payloads found for source_id. If using graphrag_ingest_langextract, use query_graph_rag_langextract.",
            }
        pairs = [{"doc_id": p.get("doc_id"), "chunk_id": p.get("chunk_id")} for p in payloads]
        score_map = {(p.get("doc_id"), p.get("chunk_id")): p.get("score") for p in payloads}

        chunks = fetch_chunks_by_pairs(pairs)
        for chunk in chunks:
            chunk["score"] = score_map.get((chunk.get("doc_id"), chunk.get("id")))
            if max_chunk_chars and chunk.get("text"):
                chunk["text"] = chunk["text"][:max_chunk_chars]

        entities = (
            fetch_entities_for_pairs(pairs, entity_types) if include_entities else []
        )

        relations = []
        related = []
        if expand_related and include_entities:
            entity_names = []
            for row in entities:
                entity_names.extend(row.get("entities") or [])
            entity_names = list(dict.fromkeys(entity_names))
            related_doc_ids: Optional[List[str]] = None
            if related_scope == "same-doc":
                if doc_id:
                    related_doc_ids = [doc_id]
                else:
                    related_doc_ids = list({p.get("doc_id") for p in pairs if p.get("doc_id")})
            related = fetch_related_chunks(entity_names, pairs, related_k, related_doc_ids)
            if max_chunk_chars and related:
                for chunk in related:
                    if chunk.get("text"):
                        chunk["text"] = chunk["text"][:max_chunk_chars]

        if include_relations:
            relations = fetch_relations_for_pairs(pairs, entity_types)

        return {
            "top_k": top_k,
            "doc_id": doc_id,
            "chunks": chunks,
            "entities": entities,
            "relations": relations,
            "related": related,
        }

    @mcp.tool()
    def list_docs(limit: int = 50) -> List[str]:
        """List available doc_id values from Neo4j."""
        with get_neo4j().session() as session:
            result = session.run(
                "MATCH (d:Document) RETURN d.id AS id ORDER BY d.id LIMIT $limit",
                limit=limit,
            )
            return [r["id"] for r in result]

    @mcp.tool()
    def list_docs_meta(limit: int = 50) -> List[Dict[str, Any]]:
        """List available docs with metadata (id, name, aliases, source, ingested_at)."""
        with get_neo4j().session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                RETURN d.id AS id,
                       d.name AS name,
                       coalesce(d.aliases, []) AS aliases,
                       d.source AS source,
                       d.ingested_at AS ingested_at
                ORDER BY d.id
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(r) for r in result]

    @mcp.tool()
    def query_graph_rag_langextract(
        query: str,
        top_k: int = 5,
        source_id: Optional[str] = None,
        include_entities: bool = True,
        include_relations: bool = True,
        expand_related: bool = True,
        related_k: int = 50,
        entity_types: Optional[List[str]] = None,
        max_passage_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query Qdrant for top-k passages with entity_ids payload, then fetch related
        entity context from Neo4j. Returns context only (no LLM generation).
        """
        embedder = get_embedder()
        q_vec = embedder.encode([query])[0].tolist()

        payloads = qdrant_search_entity_payload(
            q_vec, top_k=top_k, source_id=source_id
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
                }
            )
            entity_ids.extend(row.get("entity_ids") or [])

        entity_ids = list(dict.fromkeys(entity_ids))

        entities = fetch_entities_by_ids(entity_ids) if include_entities else []
        relations = []
        if include_relations and expand_related:
            relations = fetch_relations_by_entity_ids(
                entity_ids, entity_types, related_k
            )

        return {
            "query": query,
            "top_k": top_k,
            "source_id": source_id,
            "passages": passages,
            "entities": entities,
            "relations": relations,
        }


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
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default=os.getenv("FASTMCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FASTMCP_PORT", "8789")))
    parser.add_argument(
        "--stream-path",
        default=os.getenv("FASTMCP_STREAMABLE_HTTP_PATH", "/mcp"),
        help="Streamable HTTP path",
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
