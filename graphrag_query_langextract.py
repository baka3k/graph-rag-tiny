import argparse
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from embedding_utils import resolve_embedding_model

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()


def qdrant_search(
    client: QdrantClient, collection: str, query_vector: List[float], top_k: int
) -> List[Dict[str, Any]]:
    if hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
        )
    else:
        hits = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        ).points
    results = []
    for hit in hits:
        if hit.payload is None:
            continue
        results.append(
            {
                "score": getattr(hit, "score", None),
                "text": hit.payload.get("text"),
                "entity_ids": hit.payload.get("entity_ids") or [],
                "source_id": hit.payload.get("source_id"),
            }
        )
    return results


def fetch_related_graph(driver, entity_ids: List[str]) -> List[Dict[str, Any]]:
    if not entity_ids:
        return []
    query = """
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related
    """
    with driver.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        return [dict(record) for record in result]


def format_graph_context(subgraph: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    nodes = set()
    edges = []
    for entry in subgraph:
        entity = entry.get("e")
        related = entry.get("related")
        relation = entry.get("r")
        if not entity or not related or not relation:
            continue
        nodes.add(entity.get("name"))
        nodes.add(related.get("name"))
        edges.append(f"{entity.get('name')} {relation.get('type', 'RELATED')} {related.get('name')}")
    return {"nodes": sorted(n for n in nodes if n), "edges": edges}


def _ollama_generate(prompt: str, model_id: str, model_url: str) -> str:
    import json
    from urllib import request

    endpoint = model_url.rstrip("/") + "/api/generate"
    payload = json.dumps(
        {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")
    req = request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "") or ""


def llm_generate(prompt: str, model_id: str, model_url: str | None) -> str:
    model_id = model_id.strip()
    lower_model = model_id.lower()
    if lower_model.startswith("gemini"):
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or LANGEXTRACT_API_KEY for Gemini.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        return getattr(response, "text", "") or ""

    if lower_model.startswith("gpt-") or lower_model.startswith("o"):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using the provided graph context and citations if possible.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    if model_url:
        return _ollama_generate(prompt, model_id, model_url)

    raise RuntimeError(f"Unsupported model_id for generation: {model_id}")


def build_generation_prompt(query: str, graph_context: Dict[str, List[str]], passages: List[str]) -> str:
    nodes_str = ", ".join(graph_context.get("nodes") or [])
    edges_str = "; ".join(graph_context.get("edges") or [])
    passages_str = "\n\n".join(passages)
    return textwrap.dedent(
        f"""\
        You are an assistant with access to a knowledge graph and retrieved passages.

        Nodes: {nodes_str}
        Edges: {edges_str}

        Passages:
        {passages_str}

        Answer the user question using the graph context and passages.
        Question: {query}
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--collection", default="graphrag_entities")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-pass", default=os.getenv("NEO4J_PASS", "password"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_KEY"))
    parser.add_argument("--llm-model", default=os.getenv("LANGEXTRACT_MODEL_ID", "gemini-2.5-flash"))
    parser.add_argument("--langextract-model-id", default=os.getenv("LANGEXTRACT_MODEL_ID"))
    parser.add_argument("--langextract-model-url", default=os.getenv("LANGEXTRACT_MODEL_URL"))
    args = parser.parse_args()

    _load_env()
    _set_langextract_overrides(args.langextract_model_id, args.langextract_model_url)

    model_name, local_files_only = resolve_embedding_model(
        args.embedding_model, "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedder = SentenceTransformer(model_name, local_files_only=local_files_only)
    query_vector = embedder.encode([args.query])[0].tolist()

    if args.qdrant_url:
        qdrant = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    else:
        qdrant = QdrantClient(
            host=args.qdrant_host, port=args.qdrant_port, api_key=args.qdrant_api_key
        )

    hits = qdrant_search(qdrant, args.collection, query_vector, args.top_k)
    if not hits:
        print("No Qdrant hits found.")
        return

    entity_ids = []
    passages = []
    for hit in hits:
        passages.append(hit.get("text") or "")
        entity_ids.extend(hit.get("entity_ids") or [])
    entity_ids = list(dict.fromkeys(entity_ids))

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    subgraph = fetch_related_graph(driver, entity_ids)
    graph_context = format_graph_context(subgraph)

    prompt = build_generation_prompt(args.query, graph_context, passages)
    answer = llm_generate(prompt, args.llm_model, args.langextract_model_url)

    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
def _set_langextract_overrides(model_id: str | None, model_url: str | None) -> None:
    if model_id:
        os.environ["LANGEXTRACT_MODEL_ID"] = model_id
    if model_url:
        os.environ["LANGEXTRACT_MODEL_URL"] = model_url
