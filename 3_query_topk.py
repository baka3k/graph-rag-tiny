import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

from embedding_utils import resolve_embedding_model

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--collection", default="pdf_chunks")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    parser.add_argument("--with-entities", action="store_true")
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    model_name, local_files_only = resolve_embedding_model(
        args.embedding_model, "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    q_vec = model.encode([args.query])[0].tolist()

    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    qdrant_filter = None
    if args.doc_id:
        qdrant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_id",
                    match=qmodels.MatchValue(value=args.doc_id),
                )
            ]
        )

    if hasattr(qdrant, "search"):
        hits = qdrant.search(
            collection_name=args.collection,
            query_vector=q_vec,
            limit=args.top_k,
            query_filter=qdrant_filter,
        )
    else:
        hits = qdrant.query_points(
            collection_name=args.collection,
            query=q_vec,
            limit=args.top_k,
            query_filter=qdrant_filter,
        ).points

    hit_payloads = [
        h.payload for h in hits if h.payload and h.payload.get("chunk_id") is not None
    ]
    if not hit_payloads:
        print("No chunks found.")
        return

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    with driver.session() as session:
        if args.doc_id:
            chunk_ids = [p["chunk_id"] for p in hit_payloads]
            query = (
                "MATCH (c:Chunk {doc_id: $doc_id}) WHERE c.id IN $ids "
                "RETURN c.id AS id, c.text AS text"
            )
            result = session.run(query, doc_id=args.doc_id, ids=chunk_ids)
        else:
            doc_chunk_pairs = [
                {"doc_id": p.get("doc_id"), "chunk_id": p.get("chunk_id")}
                for p in hit_payloads
            ]
            result = session.run(
                """
                UNWIND $pairs AS pair
                MATCH (c:Chunk {doc_id: pair.doc_id, id: pair.chunk_id})
                RETURN c.id AS id, c.doc_id AS doc_id, c.text AS text
                """,
                pairs=doc_chunk_pairs,
            )

        chunks = list(result)

        print("Top-k context:\n")
        for record in chunks:
            if "doc_id" in record and record["doc_id"]:
                print(f"--- doc {record['doc_id']} chunk {record['id']} ---")
            else:
                print(f"--- chunk {record['id']} ---")
            print(record["text"])
            print()

        if args.with_entities:
            if args.doc_id:
                e_result = session.run(
                    """
                    MATCH (c:Chunk {doc_id: $doc_id})-[:MENTIONS]->(e:Entity)
                    WHERE c.id IN $ids
                    RETURN c.id AS id, collect(DISTINCT e.name) AS entities
                    """,
                    doc_id=args.doc_id,
                    ids=chunk_ids,
                )
            else:
                e_result = session.run(
                    """
                    UNWIND $pairs AS pair
                    MATCH (c:Chunk {doc_id: pair.doc_id, id: pair.chunk_id})-[:MENTIONS]->(e:Entity)
                    RETURN c.id AS id, c.doc_id AS doc_id, collect(DISTINCT e.name) AS entities
                    """,
                    pairs=doc_chunk_pairs,
                )
            print("Entities:")
            for record in e_result:
                if "doc_id" in record and record["doc_id"]:
                    print(f"doc {record['doc_id']} chunk {record['id']}: {record['entities']}")
                else:
                    print(f"chunk {record['id']}: {record['entities']}")


if __name__ == "__main__":
    main()
