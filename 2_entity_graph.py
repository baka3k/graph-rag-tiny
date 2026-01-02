import argparse
from pathlib import Path

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from entity_extractors import (
    build_spacy_pipeline,
    extract_entities_azure,
    extract_entities_gemini,
    extract_entities_gliner,
    extract_entities_langextract,
    build_gliner_model,
    parse_gliner_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--ruler-json", default=None, help="Path to EntityRuler JSON patterns")
    parser.add_argument("--gliner-model", default="urchade/gliner_mediumv2")
    parser.add_argument(
        "--gliner-labels",
        default="PERSON,ORG,PRODUCT,GPE,DATE,TECH,CRYPTO,STANDARD",
        help="Comma-separated labels or path to a text file for GLiNER",
    )
    parser.add_argument("--gliner-threshold", type=float, default=0.3)
    parser.add_argument(
        "--entity-provider",
        choices=["spacy", "gliner", "azure", "gemini", "langextract"],
        default="gliner",
        help="Entity extraction provider",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    with driver.session() as session:
        chunks = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN c.id AS id, c.text AS text",
            doc_id=args.doc_id,
        )
        if args.entity_provider == "spacy":
            nlp = build_spacy_pipeline(args.spacy_model, ruler_json=args.ruler_json)
        if args.entity_provider == "gliner":
            gliner_labels = parse_gliner_labels(args.gliner_labels)
            gliner_model = build_gliner_model(args.gliner_model)
        for record in chunks:
            chunk_id = record["id"]
            text = record["text"] or ""
            if args.entity_provider == "spacy":
                doc = nlp(text)
                entities = [{"name": ent.text, "type": ent.label_} for ent in doc.ents]
                relations = []
            elif args.entity_provider == "gliner":
                entities, relations = extract_entities_gliner(
                    text,
                    labels=gliner_labels,
                    model=args.gliner_model,
                    threshold=args.gliner_threshold,
                    gliner_model=gliner_model,
                )
            elif args.entity_provider == "azure":
                entities, relations = extract_entities_azure(text)
            elif args.entity_provider == "gemini":
                entities, relations = extract_entities_gemini(text)
            else:
                entities, relations = extract_entities_langextract(text)

            for ent in entities:
                session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id, doc_id: $doc_id})
                    MERGE (e:Entity {name: $name, type: $type})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    chunk_id=chunk_id,
                    doc_id=args.doc_id,
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

    print("Done: created Entity nodes and RELATED relationships.")


if __name__ == "__main__":
    main()
