# API Reference

The Code Explainer API provides RESTful endpoints for code explanation and retrieval services. The API is built with FastAPI and includes comprehensive OpenAPI/Swagger documentation.

## Quick Start

### Running the API Server

```bash
# Using uvicorn directly
uvicorn code_explainer.api.server:app --host 0.0.0.0 --port 8000

# Using the Makefile
make api-serve

# Using Docker
make docker-run
```

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Authentication & Rate Limiting

Currently, the API does not require authentication. Rate limiting can be configured via the `CODE_EXPLAINER_RATE_LIMIT` environment variable (default: "60/minute").

## Endpoints

### Health & Monitoring

#### GET /health
Check overall service health and component status.

**Response:**
```json
{
  "status": "ok",
  "retrieval": {
    "index_path": "data/code_retrieval_index.faiss",
    "index_exists": true,
    "corpus_exists": true,
    "index_size": 1000,
    "warmed_up": true,
    "last_error": null,
    "top_k_max": 20
  }
}
```

#### GET /version
Get the current API version.

**Response:**
```json
{
  "version": "0.3.0"
}
```

#### GET /metrics
Prometheus metrics for monitoring (requires prometheus-client).

### Code Explanation

#### POST /explain
Generate natural language explanations for Python code.

**Request:**
```json
{
  "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
  "strategy": "ast_augmented",
  "symbolic": false
}
```

**Parameters:**
- `code` (string, required): Python code to explain
- `strategy` (string, optional): Explanation strategy (`vanilla`, `ast_augmented`, `retrieval_augmented`, `execution_trace`)
- `symbolic` (boolean, optional): Include symbolic execution analysis

**Response:**
```json
{
  "explanation": "This function calculates the factorial of a number using recursion. It takes an integer n as input and returns n! (n factorial). The base case is when n is 0 or 1, returning 1. Otherwise, it multiplies n by the factorial of n-1."
}
```

#### GET /strategies
List all available explanation strategies.

**Response:**
```json
{
  "strategies": [
    "vanilla",
    "ast_augmented",
    "retrieval_augmented",
    "execution_trace"
  ]
}
```

### Code Retrieval

#### POST /retrieve
Find similar code snippets using various retrieval methods.

**Request:**
```json
{
  "code": "def recursive_function(n):",
  "k": 3,
  "method": "hybrid",
  "alpha": 0.5,
  "use_reranker": false,
  "use_mmr": false,
  "rerank_top_k": 20,
  "mmr_lambda": 0.5
}
```

**Parameters:**
- `code` (string, required): Query code snippet
- `k` (integer, optional): Number of results (1-50, default: 3)
- `method` (string, optional): Retrieval method (`faiss`, `bm25`, `hybrid`)
- `alpha` (float, optional): Hybrid weight for FAISS (0.0-1.0, default: 0.5)
- `use_reranker` (boolean, optional): Use cross-encoder reranking
- `use_mmr` (boolean, optional): Use MMR for diversity
- `rerank_top_k` (integer, optional): Candidates to rerank (1-100)
- `mmr_lambda` (float, optional): MMR relevance-diversity trade-off (0.0-1.0)

**Response:**
```json
{
  "similar_codes": [
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
  ],
  "metadata": {
    "enhanced": false
  }
}
```

#### POST /retrieve/enhanced
Enhanced retrieval with advanced features and detailed metadata.

**Request:**
```json
{
  "query": "recursive mathematical function",
  "k": 5,
  "method": "hybrid",
  "alpha": 0.7,
  "use_reranker": true,
  "use_mmr": true,
  "rerank_top_k": 50,
  "mmr_lambda": 0.3
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "def factorial(n): ...",
      "score": 0.95,
      "rerank_score": 0.87,
      "metadata": {
        "source": "example.py",
        "line_start": 10,
        "line_end": 15
      }
    }
  ],
  "query": "recursive mathematical function",
  "method_used": "hybrid",
  "total_results": 5
}
```

#### GET /retrieval/health
Detailed retrieval service health check.

**Response:**
```json
{
  "index_path": "data/code_retrieval_index.faiss",
  "index_exists": true,
  "corpus_exists": true,
  "index_size": 1000,
  "warmed_up": true,
  "last_error": null,
  "top_k_max": 20
}
```

#### GET /retrieval/stats
Runtime statistics of the retrieval service.

**Response:**
```json
{
  "loaded": true,
  "index_loaded": true,
  "corpus_size": 1000,
  "faiss_ntotal": 1000,
  "bm25_built": true,
  "index_path": "data/code_retrieval_index.faiss",
  "top_k_max": 20
}
```

## Configuration

The API can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODE_EXPLAINER_MODEL_PATH` | `./results` | Path to model directory |
| `CODE_EXPLAINER_CONFIG_PATH` | `configs/default.yaml` | Path to configuration file |
| `CODE_EXPLAINER_INDEX_PATH` | `data/code_retrieval_index.faiss` | Path to retrieval index |
| `CODE_EXPLAINER_RETRIEVAL_WARMUP` | `false` | Warm up retrieval on startup |
| `CODE_EXPLAINER_RETRIEVAL_TOPK_MAX` | `20` | Maximum k for retrieval |
| `CODE_EXPLAINER_RATE_LIMIT` | `60/minute` | API rate limit |

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error
- `503`: Service Unavailable (retrieval service not ready)

Error responses include a `detail` field with a descriptive message.

## Examples

### Basic Code Explanation

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello(name): return f\"Hello, {name}!\""}'
```

### Advanced Retrieval

```bash
curl -X POST http://localhost:8000/retrieve/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sorting algorithm",
    "k": 3,
    "method": "hybrid",
    "use_reranker": true,
    "use_mmr": true
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Docker Deployment

```bash
# Build the image
make docker-build

# Run with default settings
make docker-run

# Run in development mode (with hot reload)
make docker-dev
```

## Monitoring

The API includes built-in monitoring:

- **Prometheus metrics** at `/metrics`
- **Request tracking** with latency histograms
- **Health checks** for all components
- **Structured logging** with request IDs

Metrics include:
- Total requests by endpoint and status
- Request latency percentiles
- Retrieval operation counts and errors
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
