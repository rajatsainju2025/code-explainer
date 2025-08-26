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
 - POST /retrieve -> {"matches": ["..."], "count": N}

POST /explain payload:
```json
{
  "code": "print(1+2)",
  "strategy": "ast_augmented"
}
```

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
  "top_k": 3
}
```

Curl example:
```bash
curl -s -X POST http://localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"code": "def fib(n): ...", "index_path": "data/code_retrieval_index.faiss", "top_k": 3}' | jq
```
