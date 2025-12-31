# Install
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

# Run
```
python main.py
```

## Scripts

- `1_ingest_min.py`: PDF -> chunks -> Qdrant + Neo4j
- `2_entity_graph_spacy.py`: create Entities + MENTIONS (spaCy)
- `2_entity_graph.py`: create Entities + MENTIONS (spacy/azure)
- `3_query_topk.py`: query top-k
- `4_ingest_and_entities.py`: ingest + entities in one command
- `5_ingest_folder.py`: process all PDFs in a folder
- `0_reset_data.py`: reset data
- `mcp_graph_rag.py`: MCP server (HTTP)
- `graphrag_ingest_langextract.py`: GraphRAG ingest (LangExtract) -> Neo4j + Qdrant (payload includes entity_ids)
- `graphrag_query_langextract.py`: GraphRAG query (Qdrant + Neo4j) + generate answer

## Quick start

### 1) Ingest + Entity (single file)

```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --neo4j-pass abcd1234
```

- Use Azure OpenAI (entities + relations):
```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --entity-provider azure --neo4j-pass abcd1234
```

- Use Gemini (entities + relations):
```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --entity-provider gemini --neo4j-pass abcd1234
```

- Use LangExtract (entities + relations):
```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --entity-provider langextract --neo4j-pass abcd1234
```

- Use spaCy + EntityRuler (additional entities):
```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --entity-provider spacy --ruler-json /path/to/rules --neo4j-pass abcd1234
```

### 2) Query

- No doc_id required:

```bash
python 3_query_topk.py --query "Digital Key encryption info" --top-k 3 --neo4j-pass abcd1234
```

- With entities:

```bash
python 3_query_topk.py --query "Digital Key encryption info" --top-k 3 --neo4j-pass abcd1234 --with-entities
```

- Filter by a specific doc_id:

```bash
python 3_query_topk.py --query "Digital Key encryption info" --top-k 3 --doc-id MY_DOC_ID --neo4j-pass abcd1234
```

### 3) Ingest many PDFs in a folder

```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --neo4j-pass abcd1234
```

- Use Azure OpenAI (entities + relations):
```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --entity-provider azure --neo4j-pass abcd1234
```

- Use Gemini (entities + relations):
```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --entity-provider gemini --neo4j-pass abcd1234
```

- Use LangExtract (entities + relations):
```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --entity-provider langextract --neo4j-pass abcd1234
```

- Use spaCy + EntityRuler:
```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --entity-provider spacy --ruler-json /path/to/rules --neo4j-pass abcd1234
```

### 4) Quick reset

- Delete by doc_id:

```bash
python 0_reset_data.py --doc-id MY_DOC_ID --neo4j-pass abcd1234
```

- Delete everything:

```bash
python 0_reset_data.py --neo4j-pass abcd1234
```

## GraphRAG (2 new scripts)

### 1) Ingest -> Neo4j + Qdrant (payload includes entity_ids)
```
python graphrag_ingest_langextract.py --folder testdata --entity-provider langextract --langextract-model-id gemma2:2b --langextract-model-url http://localhost:11434 --neo4j-pass abcd1234 --min-paragraph-chars 150 --skip-llm-short   

python graphrag_ingest_langextract.py --folder testdata --entity-provider langextract --langextract-model-id gemma2:2b --langextract-model-url http://localhost:11434 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass abcd1234 --qdrant-host localhost --qdrant-port 6333 --collection graphrag_entities --embedding-model sentence- transformers/all-MiniLM-L6-v2 --min-paragraph-chars 150 --max-paragraph-chars 1200 --llm-debug --llm-retry-count 3--llm-retry-backoff 2 --skip-llm-short
```

Choose 1 of 4 inputs:

PDF:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider langextract
```

Text file:
```bash
python graphrag_ingest_langextract.py --text-file /path/to/file.txt --entity-provider gemini
```

Raw text:
```bash
python graphrag_ingest_langextract.py --raw-text "Alice works at Acme." --entity-provider spacy
```

Folder (recursive):
```bash
python graphrag_ingest_langextract.py --folder /path/to/folder --entity-provider langextract
```

Supported files: `.pdf`, `.txt`, `.md`, `.docx`, `.pptx`.

spaCy options:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider spacy --spacy-model en_core_web_sm --ruler-json /path/to/rules
```

LangExtract options (Ollama):
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider langextract --langextract-model-id gemma2:2b --langextract-model-url http://localhost:11434
```

Qdrant/Neo4j/Embedding options:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --collection graphrag_entities --embedding-model sentence-transformers/all-MiniLM-L6-v2 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass abcd1234 --qdrant-host localhost --qdrant-port 6333
```

Full command (folder + Ollama + Neo4j/Qdrant + retry/log):
```bash
python graphrag_ingest_langextract.py --folder testdata --entity-provider langextract --langextract-model-id gemma2:2b --langextract-model-url http://localhost:11434 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass abcd1234 --qdrant-host localhost --qdrant-port 6333 --collection graphrag_entities --embedding-model sentence-transformers/all-MiniLM-L6-v2 --min-paragraph-chars 150 --max-paragraph-chars 1200 --llm-debug --llm-retry-count 3 --llm-retry-backoff 2
```

Enable LLM debug logging:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider langextract --llm-debug
```
Raw output is saved to `logs/langextract_raw.log` (can be overridden via env `LLM_DEBUG_LOG_PATH`).

Retry/backoff options when parsing fails:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider langextract --llm-retry-count 3 --llm-retry-backoff 2
```

Skip paragraphs that are too short:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --min-paragraph-chars 150
```

Skip the LLM for short paragraphs (still store in Qdrant):
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --min-paragraph-chars 150 --skip-llm-short
```
If `--skip-llm-short` is enabled, short paragraphs are stored in Neo4j with label `Paragraph` (short=true).
Long paragraphs are stored as `Paragraph` and linked as `(:Paragraph)-[:HAS_ENTITY]->(:Entity)`.

### 2) Query + Generate

Basic:
```bash
python graphrag_query_langextract.py --query "How is Alice related to Acme?" --llm-model gemini-2.5-flash
```

Qdrant/Neo4j/Embedding options:
```bash
python graphrag_query_langextract.py --query "How is Alice related to Acme?" --collection graphrag_entities --top-k 5 --embedding-model sentence-transformers/all-MiniLM-L6-v2 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass abcd1234 --qdrant-host localhost --qdrant-port 6333
```

Generate with Ollama:
```bash
python graphrag_query_langextract.py --query "How is Alice related to Acme?" --llm-model gemma2:2b --langextract-model-url http://localhost:11434
```

### Suggested env (LangExtract/Gemini/OpenAI/Ollama)

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=abcd1234
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_URL=http://localhost:6333
QDRANT_KEY=your_qdrant_api_key

LANGEXTRACT_API_KEY=your_key
LANGEXTRACT_MODEL_ID=gemini-2.5-flash

# If using LangExtract + Ollama:
# LANGEXTRACT_MODEL_ID=gemma2:2b
# LANGEXTRACT_MODEL_URL=http://localhost:11434

# If using OpenAI:
# LANGEXTRACT_MODEL_ID=gpt-4o
# OPENAI_API_KEY=your_key
```

## MCP server (HTTP)

### Run server

```bash
python mcp_graph_rag.py --transport streamable-http
```

Endpoint:

```
http://127.0.0.1:8789/mcp
```

### .env (auto-loaded)

Create a `.env` file at the repo root to store settings:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=abcd1234
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=pdf_chunks
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=o4-mini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-1.5-flash
LANGEXTRACT_API_KEY=your_key
LANGEXTRACT_MODEL_ID=gemini-2.5-flash
# If using OpenAI with LangExtract:
# LANGEXTRACT_MODEL_ID=gpt-4o
# LANGEXTRACT_USE_SCHEMA_CONSTRAINTS=false
# LANGEXTRACT_FENCE_OUTPUT=true
```

Note: if you use OpenAI with LangExtract, install `langextract[openai]`.

### MCP tools

- `query_graph_rag(query, top_k, doc_id, source_id, collection, include_entities, include_relations, entity_types, expand_related, related_scope, related_k)`
- `list_docs(limit)`
- `query_graph_rag_langextract(query, top_k, source_id, collection, include_entities, include_relations, expand_related, related_k, entity_types, max_passage_chars)`
- `list_qdrant_collections()`
- `query_graph_rag_auto(query, top_k, collection, prefer, doc_id, doc_ref, source_id, include_entities, include_relations, expand_related, related_scope, related_k, entity_types, max_chunk_chars, max_passage_chars)`

Note: `query_graph_rag_langextract` uses the `entity_ids` payload from Qdrant (ingested via `graphrag_ingest_langextract.py`).

Sample calls:

```
query_graph_rag(query="Digital Key encryption info", top_k=3, include_entities=true, include_relations=true, entity_types=["ORG"], related_scope="same-doc", related_k=3)
query_graph_rag_langextract(query="Digital Key encryption info", top_k=5, source_id="my_doc", include_entities=true, include_relations=true, related_k=50, max_passage_chars=800)
list_qdrant_collections()
query_graph_rag_auto(query="Digital Key encryption info", top_k=5, collection="graphrag_entities", source_id="my_doc")
query_graph_rag_auto(query="Digital Key encryption info", top_k=5, collection="graphrag_entities", prefer="langextract", source_id="my_doc")
```

Detailed MCP examples:

```
list_qdrant_collections()
```

```
query_graph_rag(query="Digital Key encryption info", top_k=3, doc_id="MY_DOC_ID", include_entities=true, include_relations=true, related_scope="same-doc", related_k=3)
```

```
query_graph_rag_langextract(query="Digital Key encryption info", top_k=5, collection="graphrag_entities", source_id="my_doc", include_entities=true, include_relations=true, related_k=50, max_passage_chars=800)
```

```
query_graph_rag_auto(query="Digital Key encryption info", top_k=5, collection="graphrag_entities", source_id="my_doc")
```

### MCP tools comparison

| Criteria | `query_graph_rag` | `query_graph_rag_langextract` |
| --- | --- | --- |
| Qdrant payload | `doc_id`, `chunk_id` | `entity_ids`, `paragraph_id`, `source_id` |
| Matching ingest scripts | `1_ingest_min.py`, `4_ingest_and_entities.py` | `graphrag_ingest_langextract.py` |
| Returned text | Chunk from `Chunk` (Neo4j) | Passage from Qdrant payload |
| Entity/Relation | Via `MENTIONS` + `RELATED` (Neo4j) | Via `Entity` + `RELATED` based on `entity_ids` |
| Filter | `doc_id` (and `source_id` if present in payload) | `source_id` |
| Strengths | Simple and fast for the older pipeline | Fits the newer GraphRAG flow, LLM-extracted payload, flexible |
| Weaknesses | Cannot use `entity_ids` payload | Requires ingest via the newer script |
| Use case | PDF->Chunk->Neo4j/Qdrant with doc_id/chunk_id | GraphRAG entity payload + paragraph_id |

## EntityRuler JSON format
The JSON file can be a list or an object:

List:
```

Sample files (folder):
```
./rules/
```
[
  {"label": "PRODUCT", "pattern": "Digital Key"},
  {"label": "STANDARD", "pattern": [{"LOWER": "iso"}, {"IS_DIGIT": true}]}
]
```

Object:
```
{
  "patterns": [
    {"label": "PRODUCT", "pattern": "Digital Key"},
    {"label": "ORG", "pattern": "Car Connectivity Consortium"}
  ]
}
```

## Notes

- `doc_id` defaults to the filename without the `.pdf` extension (file stem).
- If your query returns no results, try removing `--doc-id` or verify the doc_id exists in Qdrant/Neo4j.
- The `page` field in `Chunk` exists only if you re-ingest with a newer version.
