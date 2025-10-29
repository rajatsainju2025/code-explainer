# Quick Start Guide

Get up and running with Code Explainer in under 5 minutes!

## Prerequisites

- Python 3.9 or higher
- pip or Poetry package manager
- (Optional) CUDA-capable GPU for faster inference

## Installation

### Option 1: Quick Install with pip

```bash
# Clone the repository
git clone https://github.com/rajatsainju2025/code-explainer.git
cd code-explainer

# Install dependencies
pip install -e .

# Or with all features
pip install -e .[all]
```

### Option 2: Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/rajatsainju2025/code-explainer.git
cd code-explainer

# Install with Poetry
poetry install --all-extras
```

### Option 3: Using Make

```bash
make install  # Auto-detects Poetry or pip
# Or
make install-dev  # Includes development tools
```

## First Run

### Command Line

Explain a simple function:

```bash
python -m code_explainer.cli explain "def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)"
```

### Python API

```python
from code_explainer import CodeExplainer

# Initialize explainer
explainer = CodeExplainer()

# Explain code
code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

explanation = explainer.explain_code(code)
print(explanation)
```

### Web UI

Start the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

### API Server

Start the REST API server:

```bash
# Development mode
make api-dev

# Or directly with uvicorn
uvicorn code_explainer.api.main:app --reload
```

Test the API:

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello(): print(\"world\")"}'
```

## Using Docker

### Quick Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

Services will be available at:
- API: http://localhost:8000
- Streamlit UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

## Configuration

### Basic Configuration

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit key settings:

```env
# Optional API key for authentication
CODE_EXPLAINER_API_KEY=your-secret-key

# Model configuration
CODE_EXPLAINER_MODEL_PATH=./results
CODE_EXPLAINER_PRECISION=fp16

# Device (cuda, mps, or cpu)
CODE_EXPLAINER_DEVICE=cuda
```

### Config Files

Edit `configs/default.yaml` to customize:

```yaml
model:
  name: Salesforce/codet5-small
  max_length: 512
  temperature: 0.7

prompt:
  strategy: vanilla  # or ast_augmented, multi_agent
```

## Explanation Strategies

Try different strategies for various use cases:

```python
# Basic explanation
explanation = explainer.explain_code(code, strategy="vanilla")

# With AST analysis
explanation = explainer.explain_code(code, strategy="ast_augmented")

# With retrieval (requires index)
explanation = explainer.explain_code(code, strategy="retrieval_augmented")

# Multi-agent collaboration
explanation = explainer.explain_code(code, strategy="multi_agent")
```

## Common Tasks

### Explain a File

```bash
python -m code_explainer.cli explain-file path/to/your/code.py
```

### Batch Processing

```python
requests = [
    {"code": "def add(a, b): return a + b"},
    {"code": "def multiply(a, b): return a * b"},
]

explanations = explainer.explain_code_batch(requests)
```

### Quality Analysis

```bash
python -m code_explainer.cli analyze-quality path/to/code.py
```

### Cache Management

```bash
# View cache statistics
python -m code_explainer.cli cache-stats

# Clear cache
python -m code_explainer.cli clear-cache
```

## Troubleshooting

### Model Not Loading

If you see "Model not found" errors:

```bash
# Download default model (if not present)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Salesforce/codet5-small')"
```

### CUDA Out of Memory

Reduce batch size or use CPU:

```bash
export CODE_EXPLAINER_DEVICE=cpu
```

Or use 8-bit quantization:

```bash
export CODE_EXPLAINER_PRECISION=8bit
```

### Import Errors

Reinstall dependencies:

```bash
pip install -e .[all]
# Or
poetry install --all-extras
```

## Next Steps

- 📖 Read the [Full Documentation](https://rajatsainju2025.github.io/code-explainer/)
- 🔧 Check out [API Usage Examples](docs/api_usage_examples.md)
- 🤝 See [Contributing Guide](CONTRIBUTING.md)
- 🧪 Try [Example Notebooks](examples/)
- 🚀 Explore [Advanced Features](docs/advanced_features.md)

## Getting Help

- 💬 [GitHub Discussions](https://github.com/rajatsainju2025/code-explainer/discussions)
- 🐛 [Report Issues](https://github.com/rajatsainju2025/code-explainer/issues)
- 📧 Email: your.email@example.com

## Quick Tips

1. **Use caching**: Identical code snippets return instantly from cache
2. **Start small**: Test with simple functions before complex code
3. **Try different strategies**: Each has strengths for different code types
4. **Monitor performance**: Use `/metrics` endpoint to track API usage
5. **Read the logs**: Set `LOG_LEVEL=DEBUG` for detailed information

Happy coding! 🎉
