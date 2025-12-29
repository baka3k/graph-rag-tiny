# Task completion checklist
- If code changes: run the relevant script manually (no test suite documented).
- Ensure `.env` is configured for Neo4j/Qdrant and chosen LLM provider.
- Verify Qdrant collection + Neo4j graph are updated by running a small query via `3_query_topk.py`.