import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from neo4j import GraphDatabase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="pdf_chunks")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    args = parser.parse_args()

    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    if args.doc_id:
        qdrant.delete(
            collection_name=args.collection,
            points_selector=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=args.doc_id),
                    )
                ]
            ),
        )
        print(f"Deleted Qdrant points for doc_id: {args.doc_id}")
    else:
        qdrant.delete_collection(collection_name=args.collection)
        print(f"Deleted Qdrant collection: {args.collection}")

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    with driver.session() as session:
        if args.doc_id:
            session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[m:MENTIONS]->(e:Entity)
                DETACH DELETE d, c, m, e
                """,
                doc_id=args.doc_id,
            )
            print(f"Deleted Neo4j nodes for doc_id: {args.doc_id}")
        else:
            session.run("MATCH (n) DETACH DELETE n")
            print("Deleted all Neo4j nodes")


if __name__ == "__main__":
    main()
