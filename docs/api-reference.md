"""API Documentation and Examples

## REST API Endpoints

### Code Explanation Endpoint

**POST** `/api/v1/explain`

Explain a code snippet with natural language description.

#### Request Body
```json
{
  "code": "def hello():\\n    return 'world'",
  "language": "python",
  "model": "codet5-base",
  "strategy": "retrieval-augmented"
}
```

#### Response
```json
{
  "code": "def hello():\\n    return 'world'",
  "explanation": "This function defines a simple method named 'hello' that returns the string 'world'.",
  "summary": "A greeting function",
  "complexity": "O(1)",
  "execution_time_ms": 125,
  "model_used": "codet5-base"
}
```

#### Error Responses
- **400 Bad Request**: Invalid code or language
- **413 Payload Too Large**: Code exceeds size limit (100KB)
- **422 Unprocessable Entity**: Invalid model configuration
- **503 Service Unavailable**: Model not loaded or device unavailable

### Batch Explanation Endpoint

**POST** `/api/v1/explain-batch`

Explain multiple code snippets in a single request.

#### Request Body
```json
{
  "items": [
    {
      "code": "x = 1",
      "language": "python",
      "model": "codet5-base"
    },
    {
      "code": "y = x + 2",
      "language": "python",
      "model": "codet5-base"
    }
  ]
}
```

#### Response
```json
{
  "results": [
    {
      "code": "x = 1",
      "explanation": "Assigns the integer value 1 to variable x.",
      "status": "success"
    },
    {
      "code": "y = x + 2",
      "explanation": "Adds 2 to the value of x and assigns the result to y.",
      "status": "success"
    }
  ],
  "total_processed": 2,
  "total_errors": 0,
  "execution_time_ms": 245
}
```

### Model Information Endpoint

**GET** `/api/v1/models`

List available explanation models and their properties.

#### Response
```json
{
  "models": [
    {
      "name": "codet5-base",
      "type": "seq2seq",
      "languages": ["python", "java", "javascript"],
      "max_tokens": 512,
      "device": "cuda",
      "status": "loaded"
    },
    {
      "name": "codebert-base",
      "type": "embedding",
      "languages": ["python", "java", "javascript", "go"],
      "device": "cpu",
      "status": "loaded"
    }
  ],
  "default_model": "codet5-base"
}
```

### Health Check Endpoint

**GET** `/api/v1/health`

Check system health and resource availability.

#### Response
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "device": "cuda",
  "device_memory_available_mb": 4096,
  "cache_items": 42,
  "uptime_seconds": 3600
}
```

## Authentication & Rate Limiting

Currently no authentication is required. Rate limiting is controlled by:
- Max request size: 100 KB per code snippet
- Max batch size: 100 items per batch request
- Timeout: 60 seconds per request

## Error Handling

All errors return a consistent format:

```json
{
  "error": {
    "type": "InvalidLanguageError",
    "message": "Language 'rust' is not supported",
    "code": "INVALID_LANGUAGE",
    "details": {
      "supported_languages": ["python", "java", "javascript"]
    }
  }
}
```

## Async Operations (Future)

Support for long-running explanations via job IDs:
- **POST** `/api/v1/explain-async` → Returns `job_id`
- **GET** `/api/v1/jobs/{job_id}` → Poll for completion
- **GET** `/api/v1/jobs/{job_id}/result` → Retrieve final result

## Versioning

API follows semantic versioning. Current version: `v1`

Breaking changes (major version bumps) will be announced with a 6-month deprecation period.

Non-breaking additions (minor version bumps) may be released frequently.
"""
