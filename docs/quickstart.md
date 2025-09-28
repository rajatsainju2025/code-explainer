# Quick Start Guide

Get up and running with Code Explainer in minutes!

## Basic Usage

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

explanation = explainer.explain(code)
print(explanation)
```

### Enhanced Strategies

```python
# Use enhanced RAG for better explanations
explainer = CodeExplainer(strategy="enhanced_rag")

# Use AST-augmented analysis
explainer = CodeExplainer(strategy="ast_augmented")

# Use retrieval-augmented generation
explainer = CodeExplainer(strategy="retrieval_augmented")

# Use execution trace for runtime analysis
explainer = CodeExplainer(strategy="execution_trace")

# Multi-agent analysis
explainer = CodeExplainer(strategy="multi_agent")
```

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
