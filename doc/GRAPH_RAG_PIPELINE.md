# Graph-RAG Minimal Pipeline (PDF -> Top-k Context)

## Muc tieu
Dựng nhanh pipeline Graph-RAG để test với 1 PDF tiếng Anh, trả về top-k context. Không cần LLM cho trả lời.

## So do pipeline
1) Ingest PDF -> text
2) Chunking (text -> chunks)
3) Embedding chunks -> vectors
4) Luu vectors vao Qdrant
5) Luu nodes/relations vao Neo4j
6) Query: embed query -> Qdrant top-k -> pull chunk text (va optional graph context) -> output

## Checklist theo tung buoc

### 0) Prerequisites
- [ ] Neo4j dang chay (bolt://localhost:7687)
- [ ] Qdrant dang chay (http://localhost:6333)
- [ ] Python venv + packages (pypdf, sentence-transformers, qdrant-client, neo4j)
- [ ] Co 1 file PDF de test

### 1) Ingest PDF
- [ ] Doc PDF -> text
- [ ] Giu metadata: doc_id, page, source path

### 2) Chunking
- [ ] Chia text thanh cac doan 500-1,000 ky tu
- [ ] Gan chunk_id, page, doc_id
- [ ] Luu chunk_text raw

### 3) Embedding
- [ ] Dung sentence-transformers (hoac embedding model khac)
- [ ] Moi chunk -> 1 vector

### 4) Qdrant
- [ ] Tao collection (size = dim embedding)
- [ ] Upsert vectors + payload {chunk_id, doc_id, page}

### 5) Neo4j Graph
- [ ] Tao node Document {id, name}
- [ ] Tao node Chunk {id, text, page}
- [ ] Tao quan he (Document)-[:HAS_CHUNK]->(Chunk)

### 6) Entity graph (optional, khong can LLM)
- [ ] Dung spaCy NER -> entity list
- [ ] Tao node Entity {name, type}
- [ ] Tao quan he (Chunk)-[:MENTIONS]->(Entity)

### 7) Query top-k
- [ ] Embed query
- [ ] Qdrant search top-k
- [ ] Lay chunk text tu Neo4j bang chunk_id
- [ ] Optional: expand graph 1 hop qua Entity

### 8) Output
- [ ] Tra ve top-k context
- [ ] (Khong can LLM) neu chi can raw context

## Khi nao can LLM?
- Extract entity/relations phuc tap (thay vi NER co san)
- Sinh cau tra loi / tom tat / reasoning

## Minimal MVP (tap trung nhat)
- Chi can: PDF -> chunk -> embed -> Qdrant -> top-k
- Neo4j chi can Document + Chunk
- Entity graph co the bo qua o vong 1

