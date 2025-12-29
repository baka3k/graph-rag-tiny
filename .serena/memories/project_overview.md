# graph-rag-tiny overview
- Purpose: GraphRAG pipeline for PDFs using Qdrant (vector store) + Neo4j (graph) with entity extraction (spaCy/Azure/Gemini/LangExtract) and query.
- Tech stack: Python, Neo4j, Qdrant, spaCy, sentence-transformers, pydantic, openai, google-generativeai, langextract, fastmcp, pypdf, python-dotenv.
- Structure: root scripts for ingestion/query/reset; rules/ for EntityRuler JSON; doc/ and testdata/ folders; .env-based config.
- Entry points: scripts like 0_reset_data.py, 1_ingest_min.py, 2_entity_graph*.py, 3_query_topk.py, 4_ingest_and_entities.py, 5_ingest_folder.py, mcp_graph_rag.py.