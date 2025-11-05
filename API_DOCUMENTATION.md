# API Documentation

## Overview

The Code Explainer provides a comprehensive API for understanding and explaining code through multiple strategies.

## Quick Start

### Basic Usage

```python
from src.code_explainer.api_simple import explain_code

# Simple explanation
result = explain_code(
    code="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    language="python"
)
print(result)
```

### Batch Processing

```python
from src.code_explainer.api_simple import batch_explain

code_samples = [
    "def add(a, b): return a + b",
    "class Calculator: pass"
]

results = batch_explain(code_samples, language="python")
for result in results:
    print(result)
```

### Strategy-Specific Explanation

```python
from src.code_explainer.api_simple import explain_with_strategy

# Use specific explanation strategy
result = explain_with_strategy(
    code="x = [i**2 for i in range(10)]",
    strategy="pattern_recognition",
    language="python"
)
```

## API Reference

### Core Functions

#### `explain_code(code: str, language: str = "python") -> Dict[str, Any]`

Explain source code with automatic strategy selection.

**Parameters:**
- `code` (str): Source code to explain
- `language` (str): Programming language ('python', 'java', 'javascript', etc.)

**Returns:**
- Dict with keys: 'explanation', 'confidence', 'strategy', 'tokens'

**Example:**
```python
result = explain_code("x = [1, 2, 3]", language="python")
```

#### `batch_explain(codes: List[str], language: str = "python") -> List[Dict]`

Explain multiple code snippets efficiently.

**Parameters:**
- `codes` (List[str]): List of code snippets
- `language` (str): Programming language

**Returns:**
- List of explanation dictionaries

**Example:**
```python
results = batch_explain(["x=1", "y=2"], language="python")
```

#### `explain_with_strategy(code: str, strategy: str, language: str = "python") -> Dict[str, Any]`

Explain code using a specific strategy.

**Parameters:**
- `code` (str): Source code
- `strategy` (str): Strategy name ('nlp', 'pattern_recognition', 'bytecode', 'ast')
- `language` (str): Programming language

**Returns:**
- Explanation dictionary with strategy-specific details

**Example:**
```python
result = explain_with_strategy(
    code="def foo(): pass",
    strategy="ast",
    language="python"
)
```

## Configuration

### Environment Variables

```bash
# Model configuration
CODE_EXPLAINER_MODEL=codellama-7b
CODE_EXPLAINER_DEVICE=cuda  # or cpu

# API configuration
CODE_EXPLAINER_MAX_CODE_LENGTH=10000
CODE_EXPLAINER_TIMEOUT=30

# Logging
CODE_EXPLAINER_LOG_LEVEL=INFO
CODE_EXPLAINER_LOG_DIR=/var/log/code-explainer
```

### Configuration File

```yaml
# config.yaml
model:
  name: "codellama-7b"
  device: "cuda"
  
api:
  max_code_length: 10000
  timeout: 30
  
logging:
  level: "INFO"
  directory: "./logs"
```

## Error Handling

### Custom Exceptions

```python
from src.code_explainer.utils.security import ValidationError

try:
    result = explain_code(code="", language="python")
except ValidationError as e:
    print(f"Validation error: {e.field_name} - {e.message}")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValidationError` | Invalid input | Check code format and language |
| `TimeoutError` | Processing too slow | Use simpler code or increase timeout |
| `ModelError` | Model not loaded | Check model path and memory |
| `SecurityError` | Malicious input detected | Review input validation |

## Performance

### Optimization Tips

1. **Batch Processing**: Use `batch_explain()` for multiple codes
2. **Language Specification**: Specify language explicitly for accuracy
3. **Caching**: Results are automatically cached
4. **Resource Management**: Use `memory.optimize_memory()` for long-running services

### Performance Metrics

```python
from src.code_explainer.utils.performance import get_performance_monitor

monitor = get_performance_monitor()
stats = monitor.get_stats("explain_time")
print(f"Average time: {stats['avg']:.3f}s")
```

## Advanced Usage

### Custom Model Configuration

```python
from src.code_explainer.utils.config_manager import ConfigManager

config = ConfigManager()
config.set("model.name", "starcoder-15b")
config.set("model.device", "cuda:0")
```

### Logging and Monitoring

```python
from src.code_explainer.utils.logging_utils import get_logger

logger = get_logger("my_app")
logger.info("Code explanation started")
```

### Security Validation

```python
from src.code_explainer.utils.security import sanitize_code_input

safe_code = sanitize_code_input(user_input, max_length=5000)
result = explain_code(safe_code)
```

## Examples

### Example 1: CLI Usage

```bash
# Simple explanation
code-explainer explain "x = [1, 2, 3]"

# With language specification
code-explainer explain --language python "def add(a, b): return a + b"

# With strategy
code-explainer explain --strategy ast "class MyClass: pass"
```

### Example 2: Web API (Streamlit)

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` and use the web interface.

### Example 3: Programmatic Usage

```python
from src.code_explainer.model.core import CodeExplainer

explainer = CodeExplainer(model_name="codellama-7b")
explanation = explainer.explain(
    code="x = sum([1, 2, 3])",
    language="python"
)
print(explanation)
```

## Integration Guide

### Django Integration

```python
from django.http import JsonResponse
from src.code_explainer.api_simple import explain_code

def explain_view(request):
    code = request.POST.get('code')
    result = explain_code(code)
    return JsonResponse(result)
```

### Flask Integration

```python
from flask import Flask, request, jsonify
from src.code_explainer.api_simple import explain_code

app = Flask(__name__)

@app.route('/explain', methods=['POST'])
def explain():
    code = request.json.get('code')
    result = explain_code(code)
    return jsonify(result)
```

### FastAPI Integration

```python
from fastapi import FastAPI
from src.code_explainer.api_simple import explain_code

app = FastAPI()

@app.post("/explain")
async def explain_endpoint(code: str):
    return explain_code(code)
```

## Troubleshooting

### Issue: "Model not found"
**Solution**: Download model or specify correct model path
```python
config.set("model.path", "/path/to/model")
```

### Issue: "Out of memory"
**Solution**: Use memory optimization
```python
from src.code_explainer.utils.memory import optimize_memory
optimize_memory()
```

### Issue: "Slow performance"
**Solution**: Enable caching and batch processing
```python
results = batch_explain(codes, language="python")
```

## Support

- **Documentation**: See README.md
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@code-explainer.dev

## Version History

- **v2.0.0**: Added enhanced caching, memory optimization, performance monitoring
- **v1.5.0**: Added security validation, unified logging
- **v1.0.0**: Initial release
