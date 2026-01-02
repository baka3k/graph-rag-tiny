import argparse

from neo4j import GraphDatabase


INDEX_STATEMENTS = [
    "CREATE INDEX chunk_doc_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id, c.id)",
    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX document_id_idx IF NOT EXISTS FOR (d:Document) ON (d.id)",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Neo4j indexes for GraphRAG.")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    with driver.session() as session:
        for stmt in INDEX_STATEMENTS:
            session.run(stmt)
            print(f"Applied: {stmt}")
    driver.close()


if __name__ == "__main__":
    main()
