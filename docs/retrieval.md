# Retrieval: FAISS, BM25, and Hybrid

The project supports three retrieval modes:
- faiss: vector similarity using SentenceTransformers
- bm25: lexical similarity using rank-bm25
- hybrid: linear fusion of normalized BM25 and FAISS similarities (alpha toward FAISS)

## Build an index

```bash
cx build-index --config configs/default.yaml --output-path data/code_retrieval_index.faiss
```

## Query an index

```bash
cx query-index --index data/code_retrieval_index.faiss --top-k 3 --method hybrid --alpha 0.5 "def add(a,b): return a+b"
```

## API
- POST /retrieve accepts: code, index_path, top_k, method, alpha
- GET /retrieval/stats returns runtime info

## Notes
- BM25 is built lazily from the loaded corpus (.corpus.json stored next to the FAISS index)
- If prometheus-client or slowapi are not installed, metrics and rate limiting gracefully no-op
