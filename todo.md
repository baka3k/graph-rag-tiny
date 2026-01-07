### 1) Fusion Strategies (Vector + Graph) – Principles and Variants

- **Late fusion (recommended baseline):** Perform vector top‑k retrieval first, then expand the graph using retrieved entity_ids. This approach is easy to implement, stable, and less prone to semantic drift from the original query.
- **Early fusion:** Use the graph to expand the query before vector retrieval. Improves recall but may introduce noise if the entity linking is incorrect.
- **Two‑stage fusion:** Conduct vector top‑k retrieval → graph expansion → reranking. Trades latency for improved result quality.
- **Entity‑type gating:** Expand the graph only for important entity types (e.g., SECURITY, PROTOCOL, DEVICE, ORG). Reduces noise.
- **Graph expansion depth:**
  - _Depth 1:_ Safe (minimal drift), suitable for precise QA.
  - _Depth 2+:_ Enhances recall, suitable for discovery and exploration.
- **Adaptive expansion:** Trigger expansion only when score or certainty is high (e.g., top‑1 score above threshold, or entity appears ≥2 times).
- **Score fusion:** Combine vector similarity scores with graph relevance metrics (e.g., node centrality, number of relations, path length) for final reranking.
- **Evidence packing:** Separate “evidence” (text passages) from “graph facts” (entities/relations) in the constructed prompt.

**Strategy selection guidelines:**

- When **accuracy** is prioritized → Late fusion + Depth 1 + Reranking.
- When **recall and exploration** are prioritized → Early fusion + Depth 2 + Entity‑type gating.
- When **latency** is prioritized → Late fusion + No reranking + Limited related_k.

---

### 2) End‑to‑End Pipeline (Practical Recommendations)

**A. Ingestion**

1. Chunk text → generate embeddings → store in Qdrant.
2. Perform entity and relation extraction → store in Neo4j.
3. _(Optional)_ Normalize entity names and apply merge policies.

**B. Retrieval**  
4. Conduct vector top‑k search in Qdrant (retrieving passages + entity*ids).  
5. *(Optional)\_ Filter results by source_id or domain.

**C. Graph Expansion**  
6. Extract entities from `entity_ids`.  
7. Expand relations based on:

- Entity types (gating).
- Number of related items (`related_k`).
- Expansion depth (start with depth 1, escalate to 2 when necessary).

**D. Reranking**  
8. Define candidates = {passages + graph facts}.  
9. Apply reranking using a cross‑encoder or LLM (when available):

- Prioritize passages containing key entities.
- Prefer short and high‑confidence relations.

**E. Prompt Assembly**  
10. **Evidence block:** top passages (concise, with provenance).  
11. **Graph block:** entities + relations (with provenance).  
12. **Query + Instruction:** clearly state reliance on the provided evidence.

**F. Answer Generation**  
13. The LLM generates answers with citations that reference both passages and graph facts.
