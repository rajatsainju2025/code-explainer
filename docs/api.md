# API Reference

The Code Explainer API provides RESTful endpoints for code explanation, retrieval, and advanced AI analysis services. The API is built with FastAPI and includes comprehensive OpenAPI/Swagger documentation with both v1 (legacy) and v2 (enhanced) endpoints.

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

## API Versions

### v1 Endpoints (Legacy)
Basic code explanation and retrieval functionality.

### v2 Endpoints (Enhanced)
Advanced features including:
- **Performance Monitoring**: Real-time metrics and memory usage
- **Security Validation**: Input sanitization and threat detection
- **Batch Processing**: Efficient multi-code explanation
- **Model Optimization**: Dynamic quantization and inference tuning
- **Rate Limiting**: Built-in request throttling
- **Health Monitoring**: Comprehensive system status

## Authentication & Security

### Rate Limiting
- Configurable via `CODE_EXPLAINER_RATE_LIMIT` environment variable (default: "60/minute")
- Sliding window algorithm prevents abuse
- Automatic cleanup of expired request records

### Input Validation
- AST-based security scanning for dangerous patterns
- Automatic detection of potentially unsafe imports and functions
- Configurable strictness levels

### Security Auditing
- Comprehensive event logging for security-related actions
- Request tracking with unique identifiers
- Configurable audit retention policies

## Endpoints

### Health & Monitoring

#### GET /health (v1)
Basic service health check.

#### GET /api/v2/health (v2)
Enhanced health check with detailed system information.

**Response:**
```json
{
  "status": "ok",
  "retrieval": {
    "index_path": "data/code_retrieval_index.faiss",
    "index_exists": true,
    "corpus_exists": true
  },
  "version": "1.0.0",
  "performance": {
    "memory_mb": {"cpu": 245.6, "gpu": 1024.8},
    "device": "cuda:0",
    "model_loaded": true
  },
  "security": {
    "rate_limiting_enabled": true,
    "input_validation_enabled": true,
    "security_monitoring_enabled": true
  }
}
```

#### GET /api/v2/performance
Get comprehensive performance metrics and system statistics.

**Response:**
```json
{
  "performance_report": "System Performance Report...\nMemory Usage: CPU: 245MB, GPU: 1024MB...",
  "timestamp": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Security & Validation

#### POST /api/v2/validate-security
Validate code for security risks without generating explanation.

**Request:**
```json
{
  "code": "import os; os.system('ls')"
}
```

**Response:**
```json
{
  "safe": false,
  "warnings": [
    "Potentially dangerous import detected: os",
    "Potentially dangerous function call detected: system"
  ],
  "code_length": 25,
  "validation_timestamp": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Code Explanation

#### POST /explain (v1)
Basic code explanation endpoint.

#### POST /api/v2/secure-explain (v2)
Secure explanation with rate limiting and input validation.

**Request:**
```json
{
  "code": "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
  "strategy": "vanilla"
}
```

**Response:**
```json
{
  "explanation": "This function implements the Fibonacci sequence recursively...",
  "code_length": 78,
  "strategy": "vanilla",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### POST /api/v2/batch-explain (v2)
Batch explanation for multiple code snippets.

**Request:**
```json
{
  "codes": [
    "def add(a, b): return a + b",
    "print('hello world')"
  ],
  "strategy": "vanilla",
  "max_length": 512
}
```

**Response:**
```json
{
  "explanations": [
    "This function adds two numbers and returns the result.",
    "This prints 'hello world' to the console."
  ],
  "batch_size": 2,
  "total_code_length": 45,
  "strategy": "vanilla",
  "batch_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Model Optimization

#### POST /api/v2/optimize-model
Apply dynamic model optimizations.

**Request:**
```json
{
  "enable_quantization": true,
  "quantization_bits": 8,
  "enable_gradient_checkpointing": false,
  "optimize_for_inference": true,
  "optimize_tokenizer": true
}
```

**Response:**
```json
{
  "optimizations_applied": {
    "quantization": "8-bit quantization applied successfully",
    "inference_optimization": true,
    "tokenizer_optimization": true
  },
  "optimization_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Retrieval & Search

#### GET /strategies
Get available explanation strategies.

#### POST /explain (with retrieval)
Code explanation with retrieval-augmented generation.

### Metrics & Monitoring

#### GET /metrics (Prometheus)
Prometheus-compatible metrics endpoint (when enabled).

**Example metrics:**
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{method="POST",endpoint="/explain"} 150

# HELP retrieval_duration_seconds Retrieval request duration
# TYPE retrieval_duration_seconds histogram
retrieval_duration_seconds_bucket{le="0.1"} 45
retrieval_duration_seconds_bucket{le="0.5"} 120
```

## Error Handling

The API uses standard HTTP status codes with detailed error messages:

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| `200` | Success | - |
| `400` | Bad Request | Invalid input, security violation, rate limit exceeded |
| `404` | Not Found | Unknown endpoint |
| `405` | Method Not Allowed | Wrong HTTP method |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server error, model failure |
| `503` | Service Unavailable | Model not loaded, retrieval service down |

**Error Response Format:**
```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODE_EXPLAINER_RATE_LIMIT` | `60/minute` | API rate limiting |
| `CODE_EXPLAINER_MODEL_PATH` | `./results` | Path to trained model |
| `CODE_EXPLAINER_CONFIG_PATH` | `configs/default.yaml` | Configuration file path |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `CODE_EXPLAINER_PRECISION` | `fp32` | Model precision (fp32, fp16, bf16, 8bit) |

### Model Optimization

The API supports dynamic model optimization:

- **Quantization**: 4-bit and 8-bit weight quantization for reduced memory usage
- **Gradient Checkpointing**: Memory-efficient training with recomputation
- **Inference Optimization**: TorchScript compilation and CUDA graph optimization
- **Tokenizer Optimization**: Vocabulary pruning and fast tokenization

## Examples

### Basic Explanation
```bash
curl -X POST "http://localhost:8000/explain" \
     -H "Content-Type: application/json" \
     -d '{"code": "def hello(): print(\"Hello, World!\")"}'
```

### Secure Validation
```bash
curl -X POST "http://localhost:8000/api/v2/validate-security" \
     -H "Content-Type: application/json" \
     -d '{"code": "import os; os.system(\"ls\")"}'
```

### Batch Processing
```bash
curl -X POST "http://localhost:8000/api/v2/batch-explain" \
     -H "Content-Type: application/json" \
     -d '{"codes": ["x = 1", "y = 2"], "strategy": "vanilla"}'
```

### Performance Monitoring
```bash
curl "http://localhost:8000/api/v2/performance"
```

## Performance Considerations

- **Batch Processing**: Use `/api/v2/batch-explain` for multiple codes (up to 10x faster)
- **Caching**: Automatic explanation caching reduces latency for repeated requests
- **Async Processing**: Non-blocking operations for high-throughput scenarios
- **Memory Management**: Automatic GPU memory monitoring and cleanup
- **Rate Limiting**: Prevents resource exhaustion under load

## Security Best Practices

- Always validate input before processing
- Use secure endpoints (`/api/v2/*`) for production
- Monitor rate limiting and security events
- Keep dependencies updated
- Use environment-specific configurations
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
