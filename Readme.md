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
