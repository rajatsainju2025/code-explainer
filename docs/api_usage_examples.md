# API Usage Examples

This document provides comprehensive examples for using the Code Explainer API.

## Table of Contents

- [Authentication](#authentication)
- [Basic Code Explanation](#basic-code-explanation)
- [Batch Processing](#batch-processing)
- [Health Checks](#health-checks)
- [Metrics and Monitoring](#metrics-and-monitoring)
- [Python Client](#python-client)
- [cURL Examples](#curl-examples)

## Authentication

The API supports optional authentication using API keys. Include your API key in the `X-API-Key` header.

```bash
# Set your API key (optional)
export API_KEY="your-secret-key-here"
```

## Basic Code Explanation

### Simple Function Explanation

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    "strategy": "vanilla"
  }'
```

Response:
```json
{
  "explanation": "This function calculates the factorial...",
  "strategy": "vanilla",
  "processing_time": 0.1234,
  "model_name": "codet5-small"
}
```

### With Maximum Length

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
    "max_length": 256,
    "strategy": "ast_augmented"
  }'
```

### Different Strategies

Available strategies:
- `vanilla`: Basic explanation
- `ast_augmented`: Enhanced with AST analysis
- `multi_agent`: Multiple agent collaboration
- `retrieval_augmented`: Uses RAG for better context

```bash
# AST-augmented explanation
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None",
    "strategy": "ast_augmented"
  }'
```

## Batch Processing

Process multiple code snippets in one request:

```bash
curl -X POST "http://localhost:8000/explain/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "requests": [
      {
        "code": "def add(a, b):\n    return a + b",
        "strategy": "vanilla"
      },
      {
        "code": "def multiply(a, b):\n    return a * b",
        "strategy": "vanilla"
      }
    ]
  }'
```

## Health Checks

### Basic Health Check

```bash
curl "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "version": "0.3.0",
  "model_loaded": true,
  "retrieval_ready": false
}
```

### Version Information

```bash
curl "http://localhost:8000/version"
```

Response:
```json
{
  "code_explainer_version": "0.3.0",
  "author": "Rajat Sainju",
  "python_version": "3.11.5",
  "torch_version": "2.0.1",
  "transformers_version": "4.30.2",
  "device": "mps"
}
```

## Metrics and Monitoring

### Performance Metrics

```bash
curl "http://localhost:8000/metrics" \
  -H "X-API-Key: ${API_KEY}"
```

Response:
```json
{
  "total_requests": 1234,
  "average_response_time": 0.1567,
  "cache_hit_rate": 0.75,
  "model_inference_time": 0.0892
}
```

### Prometheus Metrics

```bash
curl "http://localhost:8000/prometheus"
```

Returns metrics in Prometheus format for scraping.

## Python Client

### Installation

```bash
pip install requests
```

### Basic Usage

```python
import requests

class CodeExplainerClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def explain(self, code, strategy="vanilla", max_length=None):
        """Explain a code snippet."""
        payload = {
            "code": code,
            "strategy": strategy
        }
        if max_length:
            payload["max_length"] = max_length
        
        response = requests.post(
            f"{self.base_url}/explain",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def health(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Usage
client = CodeExplainerClient(api_key="your-key")

# Explain code
result = client.explain(
    code="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    strategy="ast_augmented"
)
print(result["explanation"])

# Check health
health = client.health()
print(f"Status: {health['status']}")
```

### Async Client

```python
import aiohttp
import asyncio

class AsyncCodeExplainerClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    async def explain(self, code, strategy="vanilla"):
        """Explain code asynchronously."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/explain",
                json={"code": code, "strategy": strategy},
                headers=self.headers
            ) as response:
                return await response.json()

# Usage
async def main():
    client = AsyncCodeExplainerClient()
    result = await client.explain("def hello(): print('world')")
    print(result["explanation"])

asyncio.run(main())
```

## cURL Examples

### With Request ID

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-unique-id-12345" \
  -d '{"code": "print(\"Hello\")"}'
```

### Configuration Info

```bash
curl "http://localhost:8000/config" \
  -H "X-API-Key: ${API_KEY}"
```

### Model Reload (Admin Only)

```bash
curl -X POST "http://localhost:8000/admin/reload" \
  -H "X-API-Key: ${API_KEY}"
```

## Error Handling

All errors return JSON with the following structure:

```json
{
  "error": "Error Type",
  "detail": "Detailed error message",
  "request_id": "unique-request-id"
}
```

Common status codes:
- `400`: Bad Request (validation errors)
- `401`: Unauthorized (invalid API key)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

## Rate Limiting

The API may enforce rate limits. Check response headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1609459200
```

## Best Practices

1. **Use Request IDs**: Include `X-Request-ID` for tracking
2. **Handle Errors**: Always check status codes and handle errors
3. **Batch Requests**: Use batch endpoint for multiple explanations
4. **Cache Responses**: API uses caching internally; identical requests return fast
5. **Monitor Metrics**: Use `/metrics` endpoint to track performance
6. **Set Timeouts**: Configure appropriate timeouts in your client

## Support

For issues or questions:
- GitHub Issues: https://github.com/rajatsainju2025/code-explainer/issues
- Documentation: https://rajatsainju2025.github.io/code-explainer/
