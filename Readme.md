# Install
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
# Run
```
python main.py
```

## Cac script

- `1_ingest_min.py`: PDF -> chunks -> Qdrant + Neo4j
- `2_entity_graph_spacy.py`: tao Entity + MENTIONS (spaCy)
- `2_entity_graph.py`: tao Entity + MENTIONS (spacy/azure)
- `3_query_topk.py`: query top-k
- `4_ingest_and_entities.py`: ingest + entity trong 1 lenh
- `5_ingest_folder.py`: xu ly tat ca PDF trong folder
- `0_reset_data.py`: reset data
- `mcp_graph_rag.py`: MCP server (HTTP)
- `graphrag_ingest_langextract.py`: ingest GraphRAG (LangExtract) -> Neo4j + Qdrant (payload co entity_ids)
- `graphrag_query_langextract.py`: query GraphRAG (Qdrant + Neo4j) + generate answer

## Quick start

### 1) Ingest + Entity (1 file)

```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --neo4j-pass abcd1234
```

- Dung Azure OpenAI (entity + relations):
```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --entity-provider azure --neo4j-pass abcd1234
```

- Dung Gemini (entity + relations):
```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --entity-provider gemini --neo4j-pass abcd1234
```

- Dung LangExtract (entity + relations):
```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --entity-provider langextract --neo4j-pass abcd1234
```

- Dung spaCy + EntityRuler (bo sung entity):
```bash
python 4_ingest_and_entities.py --pdf /path/to/file.pdf --entity-provider spacy --ruler-json /path/to/rules --neo4j-pass abcd1234
```

### 2) Query

- Khong can doc_id:

```bash
python 3_query_topk.py --query "Digital Key encryption info" --top-k 3 --neo4j-pass abcd1234
```

- Co entity:

```bash
python 3_query_topk.py --query "Digital Key encryption info" --top-k 3 --neo4j-pass abcd1234 --with-entities
```

- Theo doc_id cu the:

```bash
python 3_query_topk.py --query "Digital Key encryption info" --top-k 3 --doc-id MY_DOC_ID --neo4j-pass abcd1234
```

### 3) Ingest nhieu PDF trong folder

```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --neo4j-pass abcd1234
```

- Dung Azure OpenAI (entity + relations):
```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --entity-provider azure --neo4j-pass abcd1234
```

- Dung Gemini (entity + relations):
```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --entity-provider gemini --neo4j-pass abcd1234
```

- Dung LangExtract (entity + relations):
```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --entity-provider langextract --neo4j-pass abcd1234
```

- Dung spaCy + EntityRuler:
```bash
python 5_ingest_folder.py --folder /path/to/pdf_folder --entity-provider spacy --ruler-json /path/to/rules --neo4j-pass abcd1234
```

### 4) Reset nhanh

- Xoa theo doc_id:

```bash
python 0_reset_data.py --doc-id MY_DOC_ID --neo4j-pass abcd1234
```

- Xoa toan bo:

```bash
python 0_reset_data.py --neo4j-pass abcd1234
```

## GraphRAG (2 script moi)

### 1) Ingest -> Neo4j + Qdrant (payload co entity_ids)
```
python graphrag_ingest_langextract.py --folder testdata --entity-provider langextract --langextract-model-id gemma2:2b --langextract-model-url http://localhost:11434 --neo4j-pass abcd1234 --min-paragraph-chars 150 --skip-llm-short   

python graphrag_ingest_langextract.py --folder testdata --entity-provider langextract --langextract-model-id gemma2:2b --langextract-model-url http://localhost:11434 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass abcd1234 --qdrant-host localhost --qdrant-port 6333 --collection graphrag_entities --embedding-model sentence- transformers/all-MiniLM-L6-v2 --min-paragraph-chars 150 --max-paragraph-chars 1200 --llm-debug --llm-retry-count 3--llm-retry-backoff 2 --skip-llm-short
```
Chon 1 trong 4 input:

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

Folder (de quy):
```bash
python graphrag_ingest_langextract.py --folder /path/to/folder --entity-provider langextract
```

Ho tro file: `.pdf`, `.txt`, `.md`, `.docx`, `.pptx`.

Tuy chon cho spaCy:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider spacy --spacy-model en_core_web_sm --ruler-json /path/to/rules
```

Tuy chon LangExtract (Ollama):
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider langextract --langextract-model-id gemma2:2b --langextract-model-url http://localhost:11434
```

Tuy chon Qdrant/Neo4j/Embedding:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --collection graphrag_entities --embedding-model sentence-transformers/all-MiniLM-L6-v2 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass abcd1234 --qdrant-host localhost --qdrant-port 6333
```

Lenh day du (folder + Ollama + Neo4j/Qdrant + retry/log):
```bash
python graphrag_ingest_langextract.py --folder testdata --entity-provider langextract --langextract-model-id gemma2:2b --langextract-model-url http://localhost:11434 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass abcd1234 --qdrant-host localhost --qdrant-port 6333 --collection graphrag_entities --embedding-model sentence-transformers/all-MiniLM-L6-v2 --min-paragraph-chars 150 --max-paragraph-chars 1200 --llm-debug --llm-retry-count 3 --llm-retry-backoff 2
```

Bat log output LLM:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider langextract --llm-debug
```
Log raw output se duoc luu o `logs/langextract_raw.log` (co the override bang env `LLM_DEBUG_LOG_PATH`).

Tuy chon retry/backoff khi parse loi:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --entity-provider langextract --llm-retry-count 3 --llm-retry-backoff 2
```

Bo qua paragraph qua ngan:
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --min-paragraph-chars 150
```

Bo qua LLM cho paragraph ngan (van luu Qdrant):
```bash
python graphrag_ingest_langextract.py --pdf /path/to/file.pdf --min-paragraph-chars 150 --skip-llm-short
```
Neu bat `--skip-llm-short`, doan ngan se duoc luu vao Neo4j label `Paragraph` (short=true).
Doan dai se luu `Paragraph` va lien ket `(:Paragraph)-[:HAS_ENTITY]->(:Entity)`.

### 2) Query + Generate

Co ban:
```bash
python graphrag_query_langextract.py --query "How is Alice related to Acme?" --llm-model gemini-2.5-flash
```

Tuy chon Qdrant/Neo4j/Embedding:
```bash
python graphrag_query_langextract.py --query "How is Alice related to Acme?" --collection graphrag_entities --top-k 5 --embedding-model sentence-transformers/all-MiniLM-L6-v2 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass abcd1234 --qdrant-host localhost --qdrant-port 6333
```

Generate bang Ollama:
```bash
python graphrag_query_langextract.py --query "How is Alice related to Acme?" --llm-model gemma2:2b --langextract-model-url http://localhost:11434
```

### Env goi y (LangExtract/Gemini/OpenAI/Ollama)

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

# Neu dung LangExtract + Ollama:
# LANGEXTRACT_MODEL_ID=gemma2:2b
# LANGEXTRACT_MODEL_URL=http://localhost:11434

# Neu dung OpenAI:
# LANGEXTRACT_MODEL_ID=gpt-4o
# OPENAI_API_KEY=your_key
```

## MCP server (HTTP)

### Chay server

```bash
python mcp_graph_rag.py --transport streamable-http
```

Endpoint:

```
http://127.0.0.1:8789/mcp
```

### .env (tu dong load)

Tao file `.env` o root de luu thong tin:

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
# Neu dung OpenAI voi LangExtract:
# LANGEXTRACT_MODEL_ID=gpt-4o
# LANGEXTRACT_USE_SCHEMA_CONSTRAINTS=false
# LANGEXTRACT_FENCE_OUTPUT=true
```

Ghi chu: neu dung OpenAI voi LangExtract, can cai them `langextract[openai]`.

### MCP tools

- `query_graph_rag(query, top_k, doc_id, include_entities, include_relations, entity_types, expand_related, related_scope, related_k)`
- `list_docs(limit)`

Goi mau:

```
query_graph_rag(query="Digital Key encryption info", top_k=3, include_entities=true, include_relations=true, entity_types=["ORG"], related_scope="same-doc", related_k=3)
```

## EntityRuler JSON format
File JSON co the la list hoac object:

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

## Luu y

- `doc_id` mac dinh la ten file khong co duoi `.pdf` (file stem).
- Neu query khong ra ket qua, thu bo `--doc-id` hoac kiem tra doc_id dang co trong Qdrant/Neo4j.
- Truong `page` trong `Chunk` chi co neu ingest lai voi phien ban moi.
