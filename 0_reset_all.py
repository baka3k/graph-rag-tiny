import argparse
import os

from neo4j import GraphDatabase
from qdrant_client import QdrantClient


def reset_neo4j(uri: str, user: str, password: str) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()


def reset_qdrant(
    collection: str,
    url: str | None,
    host: str,
    port: int,
    api_key: str | None,
) -> None:
    if url:
        client = QdrantClient(url=url, api_key=api_key)
    else:
        client = QdrantClient(host=host, port=port, api_key=api_key)
    try:
        client.delete_collection(collection_name=collection)
    except Exception as exc:
        message = str(exc)
        if "Not found" not in message and "does not exist" not in message:
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset Neo4j data and Qdrant collection.")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-pass", default=os.getenv("NEO4J_PASS", "password"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_KEY"))
    parser.add_argument(
        "--qdrant-collection",
        default=os.getenv("QDRANT_COLLECTION", "graphrag_entities"),
    )
    args = parser.parse_args()

    print("Resetting Neo4j...")
    reset_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    print("Neo4j reset complete.")

    print("Resetting Qdrant collection...")
    reset_qdrant(
        args.qdrant_collection,
        args.qdrant_url,
        args.qdrant_host,
        args.qdrant_port,
        args.qdrant_api_key,
    )
    print("Qdrant reset complete.")


if __name__ == "__main__":
    main()
