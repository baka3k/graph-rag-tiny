# Graph-RAG Minimal Pipeline (PDF -> Top-k Context)

## Goal
Quickly build a minimal Graph-RAG pipeline to test with a single English PDF and return top-k context. No LLM is required to generate an answer.

## Pipeline overview
1) Ingest PDF -> text
2) Chunking (text -> chunks)
3) Embedding chunks -> vectors
4) Store vectors in Qdrant
5) Store nodes/relations in Neo4j
6) Query: embed query -> Qdrant top-k -> pull chunk text (and optional graph context) -> output

## Step-by-step checklist

### 0) Prerequisites
- [ ] Neo4j is running (bolt://localhost:7687)
- [ ] Qdrant is running (http://localhost:6333)
- [ ] Python venv + packages (pypdf, sentence-transformers, qdrant-client, neo4j)
- [ ] Have 1 PDF file to test

### 1) Ingest PDF
- [ ] Read PDF -> text
- [ ] Keep metadata: doc_id, page, source path

### 2) Chunking
- [ ] Split text into chunks of 500-1,000 characters
- [ ] Assign chunk_id, page, doc_id
- [ ] Store raw chunk_text

### 3) Embedding
- [ ] Use sentence-transformers (or another embedding model)
- [ ] Each chunk -> 1 vector

### 4) Qdrant
- [ ] Create collection (size = embedding dimension)
- [ ] Upsert vectors + payload {chunk_id, doc_id, page}

### 5) Neo4j Graph
- [ ] Create Document node {id, name}
- [ ] Create Chunk node {id, text, page}
- [ ] Create relationship (Document)-[:HAS_CHUNK]->(Chunk)

### 6) Entity graph (optional, no LLM required)
- [ ] Use spaCy NER -> entity list
- [ ] Create Entity node {name, type}
- [ ] Create relationship (Chunk)-[:MENTIONS]->(Entity)

### 7) Query top-k
- [ ] Embed query
- [ ] Qdrant search top-k
- [ ] Fetch chunk text from Neo4j by chunk_id
- [ ] Optional: expand the graph 1 hop via Entity

### 8) Output
- [ ] Return top-k context
- [ ] (No LLM required) if you only need raw context

## When do you need an LLM?
- Extract complex entities/relations (instead of basic NER)
- Generate answers / summaries / reasoning

## Minimal MVP (most focused)
- Only need: PDF -> chunk -> embed -> Qdrant -> top-k
- Neo4j only needs Document + Chunk
- The entity graph can be skipped in v1

