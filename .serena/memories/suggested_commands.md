# Suggested commands (Windows, PowerShell)
- Install deps: `pip install -r requirements.txt`
- Download spaCy model: `python -m spacy download en_core_web_sm`
- Run MCP server: `python mcp_graph_rag.py --transport streamable-http`
- Ingest + entities (1 file): `python 4_ingest_and_entities.py --pdf <path> --neo4j-pass <pass>`
- Query: `python 3_query_topk.py --query "..." --top-k 3 --neo4j-pass <pass>`
- Ingest folder: `python 5_ingest_folder.py --folder <path> --neo4j-pass <pass>`
- Reset: `python 0_reset_data.py --neo4j-pass <pass>`