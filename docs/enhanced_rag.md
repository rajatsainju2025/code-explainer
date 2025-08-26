# Enhanced Retrieval-Augmented Generation (RAG)

The Enhanced RAG feature provides contextually-aware code explanations by retrieving similar code snippets from a pre-built knowledge base.

## Overview

Enhanced RAG uses semantic similarity search to find relevant code examples and includes them in the prompt context, helping the model provide more accurate and contextually relevant explanations.

## How It Works

1. **Index Building**: Create a FAISS index from your training data or custom code corpus
2. **Similarity Search**: For each query, find the most similar code snippets using sentence embeddings
3. **Context Enhancement**: Include similar examples in the prompt to guide the explanation
4. **Fallback**: Gracefully falls back to vanilla prompts if retrieval fails

## Usage

### Quickstart

If you already have training data configured in `configs/enhanced.yaml`, you can build and query the index in two steps:

```bash
# 1) Build the index from your configured train file
python -m code_explainer.cli build-index \
    --config configs/enhanced.yaml \
    --output-path data/code_retrieval_index.faiss

# 2) Query the index with a code snippet (prints top matches)
python -m code_explainer.cli query-index \
    --index data/code_retrieval_index.faiss \
    --top-k 3 \
    "def merge_sort(arr): ..."
```

### Building an Index

First, build a FAISS index from your training data:

```bash
# Using the CLI
python -m code_explainer.cli build-index \
    --config configs/enhanced.yaml \
    --output-path data/code_retrieval_index.faiss

# Or programmatically
from code_explainer.retrieval import CodeRetriever

codes = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    "def bubble_sort(arr): ...",
]

retriever = CodeRetriever()
retriever.build_index(codes, save_path="my_index.faiss")
```

### Using Enhanced RAG

```python
from code_explainer.model import CodeExplainer

# Initialize with enhanced RAG configuration
explainer = CodeExplainer(
    model_path="./results",
    config_path="configs/enhanced.yaml"
)

# Explain code using enhanced RAG
code = "def quick_sort(arr): ..."
explanation = explainer.explain_code(code, strategy="enhanced_rag")
```

### CLI Usage

```bash
# Explain code with enhanced RAG
python -m code_explainer.cli explain \
    --prompt-strategy enhanced_rag \
    "def merge_sort(arr): return sorted(arr)"

# Explain a file with enhanced RAG
python -m code_explainer.cli explain-file \
    --prompt-strategy enhanced_rag \
    path/to/code.py
```

### Web Interface

The Gradio interface includes Enhanced RAG as one of the selectable prompt strategies:

```python
# Start the web server
python -m code_explainer.cli serve

# Select "enhanced_rag" from the strategy dropdown
```

### API Usage

```python
import requests

response = requests.post("http://localhost:8000/explain", json={
    "code": "def binary_search(arr, target): ...",
    "strategy": "enhanced_rag"
})

explanation = response.json()["explanation"]
```

## Configuration

Configure Enhanced RAG in your YAML config file:

```yaml
# Enhanced RAG configuration
retrieval:
  index_path: "data/code_retrieval_index.faiss"
  embedding_model: "microsoft/codebert-base"
  index_dimension: 768
  similarity_top_k: 3
  similarity_threshold: 0.7
  chunk_size: 512
  chunk_overlap: 50

prompting:
  strategy: "enhanced_rag"
```

## Configuration Options

- **index_path**: Path to the FAISS index file
- **embedding_model**: Sentence transformer model for encoding code
- **similarity_top_k**: Number of similar examples to retrieve
- **similarity_threshold**: Minimum similarity score for inclusion
- **chunk_size**: Maximum length of code chunks in the index
- **chunk_overlap**: Overlap between consecutive chunks

## Best Practices

1. **Quality Corpus**: Build your index from high-quality, well-documented code examples
2. **Domain-Specific**: Use code examples relevant to your target domain
3. **Regular Updates**: Rebuild the index as you add new training data
4. **Monitoring**: Monitor fallback rates to ensure index availability
5. **Performance**: Consider index size vs. retrieval speed trade-offs

## Performance

Enhanced RAG adds minimal latency (~100-200ms) for semantic search while significantly improving explanation quality through contextual examples.

## Troubleshooting

### Index Not Found
```
WARNING: Enhanced RAG failed: Index file not found. Falling back to vanilla prompt.
```
**Solution**: Build an index using the `build-index` command or check the `index_path` configuration.

### Memory Issues
```
ERROR: FAISS index loading failed: Out of memory
```
**Solution**: Use a smaller embedding model or reduce the corpus size.

### Poor Retrieval Quality
```
Retrieved examples seem irrelevant
```
**Solution**: 
- Increase `similarity_threshold`
- Use a more appropriate embedding model
- Improve the quality of your code corpus

## Examples

See `examples/enhanced_rag_demo.py` for a complete demonstration of building an index and using Enhanced RAG for code explanation.
