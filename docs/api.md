# API Usage (FastAPI)

Run the API:

```bash
uvicorn code_explainer.api.server:app --host 0.0.0.0 --port 8000
```

Environment:
- CODE_EXPLAINER_MODEL_PATH: path to model dir (default ./results)
- CODE_EXPLAINER_CONFIG_PATH: path to config (default configs/default.yaml)

Endpoints:
- GET /health -> {"status":"ok"}
- GET /version -> {"version": "<semver>"}
- GET /strategies -> {"strategies": ["vanilla","ast_augmented","retrieval_augmented","execution_trace"]}
- POST /explain -> {"explanation": "..."}
 - POST /retrieve -> {"matches": ["..."], "count": N, "method": "hybrid", "alpha": 0.5}
 - GET /retrieval/health -> retrieval index status
 - GET /retrieval/stats -> runtime stats of retriever
 - GET /metrics -> Prometheus metrics (if prometheus-client installed)

POST /explain payload:
```json
{
  "code": "print(1+2)",
  "strategy": "ast_augmented"
}
```

Validation:
- code: non-empty string (1+ chars)
- strategy: optional; see /strategies

Curl example:
```bash
curl -s -X POST http://localhost:8000/explain \
  -H 'Content-Type: application/json' \
  -d '{"code": "def add(a,b): return a+b", "strategy": "ast_augmented"}' | jq
```

POST /retrieve payload:
```json
{
  "code": "def merge_sort(arr): ...",
  "index_path": "data/code_retrieval_index.faiss",
  "top_k": 3,
  "method": "hybrid",  // faiss | bm25 | hybrid
  "alpha": 0.5           // hybrid weight toward FAISS similarity
}
```

Curl example:
```bash
curl -s -X POST http://localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"code": "def fib(n): ...", "index_path": "data/code_retrieval_index.faiss", "top_k": 3}' | jq
```

Validation:
- code: min length 3
- top_k: 1..CODE_EXPLAINER_RETRIEVAL_TOPK_MAX (default 20)
 - method: one of faiss|bm25|hybrid (default hybrid)
 - alpha: 0..1 (default 0.5)

Health endpoints:
```bash
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/retrieval/health | jq
curl -s http://localhost:8000/retrieval/stats | jq
```

Rate limiting:
- Configure with env CODE_EXPLAINER_RATE_LIMIT (e.g., "60/minute").
- Requires slowapi to be installed; otherwise disabled automatically.
