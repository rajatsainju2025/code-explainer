# Quick Start Guide

Get up and running with Code Explainer in minutes! This guide covers basic usage, advanced features, and production deployment.

## 🚀 Basic Usage

### Python SDK

```python
from code_explainer import CodeExplainer

# Initialize with default settings
explainer = CodeExplainer()

# Explain a simple function
code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

explanation = explainer.explain_code(code)
print(explanation)
```

### Enhanced Strategies

```python
# Use enhanced RAG for better explanations
explainer = CodeExplainer()
explanation = explainer.explain_code(code, strategy="enhanced_rag")

# Use AST-augmented analysis
explanation = explainer.explain_code(code, strategy="ast_augmented")

# Use retrieval-augmented generation
explanation = explainer.explain_code(code, strategy="retrieval_augmented")

# Use execution trace for runtime analysis
explanation = explainer.explain_code(code, strategy="execution_trace")

# Multi-agent analysis
explanation = explainer.explain_code_multi_agent(code)
```

## 🌐 REST API

### Starting the API Server

```bash
# Using uvicorn
uvicorn code_explainer.api.server:app --host 0.0.0.0 --port 8000

# Using the Makefile
make api-serve

# Using Docker
make docker-run
```

### Basic API Usage

```bash
# Health check
curl http://localhost:8000/api/v2/health

# Explain code (v1)
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello(): return \"Hello, World!\""}'

# Secure explanation (v2)
curl -X POST "http://localhost:8000/api/v2/secure-explain" \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello(): return \"Hello, World!\"", "strategy": "vanilla"}'
```

### Advanced API Features

```bash
# Security validation
curl -X POST "http://localhost:8000/api/v2/validate-security" \
  -H "Content-Type: application/json" \
  -d '{"code": "import os; os.system(\"ls\")"}'

# Batch processing
curl -X POST "http://localhost:8000/api/v2/batch-explain" \
  -H "Content-Type: application/json" \
  -d '{
    "codes": ["def add(a,b): return a+b", "print(\"hello\")"],
    "strategy": "vanilla"
  }'

# Performance monitoring
curl http://localhost:8000/api/v2/performance

# Model optimization
curl -X POST "http://localhost:8000/api/v2/optimize-model" \
  -H "Content-Type: application/json" \
  -d '{"enable_quantization": true, "optimize_for_inference": true}'
```

## ⚡ Performance Optimization

### Memory and Speed Optimization

```python
from code_explainer import CodeExplainer

explainer = CodeExplainer()

# Enable quantization for memory efficiency
explainer.enable_quantization(bits=8)  # 50% memory reduction

# Optimize for inference speed
explainer.optimize_for_inference()  # 2-3x speedup

# Check performance
report = explainer.get_performance_report()
print(report)
```

### Batch Processing

```python
# Process multiple codes efficiently
codes = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "class Calculator: def add(self, a, b): return a + b",
    "import math; print(math.pi)"
]

# Batch explanation (5-10x faster)
explanations = explainer.explain_code_batch(codes, strategy="vanilla")

# Async processing for concurrent requests
import asyncio

async def explain_async():
    tasks = [asyncio.create_task(asyncio.to_thread(explainer.explain_code, code)) for code in codes]
    return await asyncio.gather(*tasks)

results = asyncio.run(explain_async())
```

## 🔒 Security Features

### Input Validation

```python
# Automatic security validation
is_safe, warnings = explainer.validate_input_security(code)
if not is_safe:
    print(f"Security warnings: {warnings}")
else:
    explanation = explainer.explain_code(code)

# Secure explanation with rate limiting
explanation = explainer.secure_explain_code(code, strategy="vanilla")
```

### Security Configuration

```bash
# Environment variables for security
export CODE_EXPLAINER_RATE_LIMIT="60/minute"
export CODE_EXPLAINER_SECURITY_STRICT=true
export CODE_EXPLAINER_MAX_CODE_LENGTH=10000
```

## 📊 Monitoring and Metrics

### Performance Monitoring

```python
# Real-time memory monitoring
memory_stats = explainer.get_memory_usage()
print(f"Memory usage: {memory_stats}")

# Performance report
report = explainer.get_performance_report()
print(report)
```

### Prometheus Metrics

```bash
# Access metrics endpoint
curl http://localhost:8000/metrics

# Example metrics output
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{method="POST",endpoint="/explain"} 150
```

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build and run
docker build -t code-explainer .
docker run -p 8000:8000 code-explainer

# With GPU support
docker run --gpus all -p 8000:8000 code-explainer

# With custom config
docker run -v $(pwd)/config.yaml:/app/config.yaml -p 8000:8000 code-explainer
```

### Docker Compose

```yaml
version: '3.8'
services:
  code-explainer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CODE_EXPLAINER_MODEL=microsoft/CodeGPT-small-py
      - CODE_EXPLAINER_RATE_LIMIT=100/minute
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./cache:/app/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## ⚙️ Configuration

### Basic Configuration

```yaml
# config.yaml
model:
  name: "microsoft/CodeGPT-small-py"
  max_length: 512
  device: "auto"

cache:
  enabled: true
  max_size: 10000

security:
  strict_mode: true
  rate_limiting_enabled: true

performance:
  enable_quantization: false
  memory_monitoring_enabled: true
```

### Environment-Based Config

```bash
# Development
export CODE_EXPLAINER_LOG_LEVEL=DEBUG
export CODE_EXPLAINER_CACHE_ENABLED=false

# Production
export CODE_EXPLAINER_RATE_LIMIT=60/minute
export CODE_EXPLAINER_ENABLE_QUANTIZATION=true
export CODE_EXPLAINER_SECURITY_STRICT=true
```

## 🧪 Testing and Validation

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_api_v2.py -v
pytest tests/test_integration_new_features.py -v

# Run performance tests
pytest tests/test_performance_regression.py -v

# Run security tests
pytest tests/test_security.py -v
```

### Benchmarking

```bash
# Performance benchmarking
python benchmarks/benchmark_performance.py

# Memory usage testing
python benchmarks/benchmark_memory.py

# Load testing
python benchmarks/benchmark_load.py
```

## 📈 Advanced Usage

### Custom Strategies

```python
from code_explainer import CodeExplainer

explainer = CodeExplainer()

# Intelligent explanation with audience targeting
explanation = explainer.explain_code_intelligent(
    code,
    audience="beginner",
    style="tutorial",
    include_examples=True
)

# Symbolic analysis
explanation = explainer.explain_code_with_symbolic(code)

# Multi-agent collaboration
explanation = explainer.explain_code_multi_agent(code)
```

### Caching Optimization

```python
# Advanced caching configuration
explainer = CodeExplainer()

# Check cache statistics
stats = explainer.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}%")

# Pre-warm cache
common_patterns = ["def ", "class ", "import "]
for pattern in common_patterns:
    explainer.explain_code(pattern)
```

## 🚀 Production Deployment

### System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **Storage**: 10GB+ for models and cache
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional but recommended)

### Production Checklist

- [ ] Configure rate limiting and security
- [ ] Enable monitoring and metrics
- [ ] Set up log aggregation
- [ ] Configure backup and recovery
- [ ] Test high-load scenarios
- [ ] Set up health checks and alerts

### Scaling Considerations

```yaml
# Production docker-compose.yml
version: '3.8'
services:
  code-explainer:
    image: code-explainer:latest
    ports:
      - "8000:8000"
    environment:
      - CODE_EXPLAINER_MODEL=microsoft/CodeGPT-small-py
      - CODE_EXPLAINER_RATE_LIMIT=200/minute
      - CODE_EXPLAINER_ENABLE_QUANTIZATION=true
    volumes:
      - cache:/app/.cache
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

volumes:
  cache:
```

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check available models
python -c "from transformers import AutoModelForCausalLM; print(AutoModelForCausalLM.from_pretrained('microsoft/CodeGPT-small-py'))"

# Clear cache and retry
rm -rf .cache/
```

#### Memory Issues
```python
# Enable quantization
explainer.enable_quantization(bits=8)

# Check memory usage
memory = explainer.get_memory_usage()
print(f"Current memory: {memory}")
```

#### API Connection Issues
```bash
# Test API connectivity
curl http://localhost:8000/api/v2/health

# Check server logs
docker logs code-explainer

# Verify port availability
netstat -tlnp | grep 8000
```

### Getting Help

- 📖 [Documentation](https://rajatsainju2025.github.io/code-explainer/)
- 💬 [Discussions](https://github.com/rajatsainju2025/code-explainer/discussions)
- 🐛 [Issues](https://github.com/rajatsainju2025/code-explainer/issues)
- 📧 Security issues: security@code-explainer.dev

## 🎯 Next Steps

1. **Explore Advanced Features**: Try different explanation strategies and caching options
2. **API Integration**: Integrate with your applications using the REST API
3. **Performance Tuning**: Optimize for your specific use case and hardware
4. **Production Deployment**: Set up monitoring, scaling, and security
5. **Contribute**: Join the community and help improve Code Explainer!

## Command Line Interface

### Basic Commands

```bash
# Explain a file
code-explainer explain --file examples/fibonacci.py

# Use specific strategy
code-explainer explain --file mycode.py --strategy enhanced_rag

# Explain code from stdin
echo "def hello(): print('world')" | code-explainer explain --stdin
```

### Evaluation

```bash
# Run evaluations on standard datasets
code-explainer eval --dataset humaneval --model codet5-small

# Custom evaluation
code-explainer eval --dataset my_dataset.json --output results.json
```

### Security Checks

```bash
# Check code for security issues
code-explainer security --file suspicious_code.py

# Scan directory
code-explainer security --dir ./src/
```

## Web Interfaces

### FastAPI Server

```bash
# Start the API server
make serve
# or
uvicorn src.code_explainer.api.server:app --reload

# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Streamlit App

```bash
# Start Streamlit interface
make streamlit
# or
streamlit run src/code_explainer/web/streamlit_app.py

# Access at http://localhost:8501
```

### Gradio Interface

```bash
# Start Gradio interface
make gradio
# or
python src/code_explainer/web/gradio_app.py

# Access at http://localhost:7860
```

## REST API Usage

### Basic Explanation

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b): return a + b",
    "strategy": "enhanced_rag"
  }'
```

### Enhanced Retrieval

```bash
curl -X POST "http://localhost:8000/retrieve/enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sorting algorithms",
    "top_k": 5,
    "use_reranker": true
  }'
```

## Configuration

### Environment Variables

```bash
# API settings
export CODE_EXPLAINER_MODEL="microsoft/codebert-base"
export CODE_EXPLAINER_MAX_LENGTH=512
export CODE_EXPLAINER_DEVICE="cuda"

# Database settings
export CODE_EXPLAINER_DB_URL="sqlite:///code_explainer.db"

# Monitoring
export CODE_EXPLAINER_METRICS_ENABLED=true
export CODE_EXPLAINER_LOG_LEVEL="INFO"
```

### Configuration File

Create `config.yaml`:

```yaml
model:
  name: "microsoft/codebert-base"
  max_length: 512
  device: "auto"

retrieval:
  enabled: true
  top_k: 5
  use_reranker: true
  similarity_threshold: 0.7

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

logging:
  level: "INFO"
  format: "json"
```

## Examples

### Batch Processing

```python
from code_explainer import CodeExplainer
import json

explainer = CodeExplainer(strategy="enhanced_rag")

# Process multiple files
files = ["example1.py", "example2.py", "example3.py"]
results = []

for file_path in files:
    with open(file_path, 'r') as f:
        code = f.read()

    explanation = explainer.explain(code)
    results.append({
        "file": file_path,
        "explanation": explanation,
        "metrics": explainer.get_last_metrics()
    })

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Custom Preprocessing

```python
from code_explainer import CodeExplainer
from code_explainer.preprocessing import clean_code, extract_functions

explainer = CodeExplainer()

# Custom preprocessing
code = """
# This is a messy file with comments
def my_function(x):
    # TODO: optimize this
    return x * 2

# Another function
def helper():
    pass
"""

# Clean and extract functions
cleaned = clean_code(code)
functions = extract_functions(cleaned)

# Explain each function
for func_name, func_code in functions.items():
    explanation = explainer.explain(func_code)
    print(f"\n{func_name}:\n{explanation}")
```

## Performance Tips

1. **Use GPU**: Install CUDA-enabled PyTorch for faster inference
2. **Batch Processing**: Process multiple files together for efficiency
3. **Caching**: Enable caching for repeated explanations
4. **Model Selection**: Choose appropriate model size for your use case

## Next Steps

- Explore [different strategies](strategies.md)
- Learn about [configuration options](configuration.md)
- Check the [API reference](api/rest.md)
- See [advanced examples](../examples/)
