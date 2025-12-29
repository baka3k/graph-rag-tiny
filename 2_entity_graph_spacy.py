import argparse
from neo4j import GraphDatabase
import spacy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    with driver.session() as session:
        chunks = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN c.id AS id, c.text AS text",
            doc_id=args.doc_id,
        )
        for record in chunks:
            chunk_id = record["id"]
            text = record["text"] or ""
            doc = nlp(text)

            for ent in doc.ents:
                session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id, doc_id: $doc_id})
                    MERGE (e:Entity {name: $name, type: $type})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    chunk_id=chunk_id,
                    doc_id=args.doc_id,
                    name=ent.text,
                    type=ent.label_,
                )

    print("Done: created Entity nodes and MENTIONS relationships.")


if __name__ == "__main__":
    main()
