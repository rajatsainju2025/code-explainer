"""Example Notebooks for Code Explainer

This directory contains Jupyter notebooks demonstrating Code Explainer usage.

## Notebooks

### 1. code_explainer_comprehensive_guide.ipynb
Complete guide to all Code Explainer features:
- Basic code explanation
- Batch processing
- Multi-agent consensus explanations
- Retrieval-augmented generation
- Input sanitization
- Cache management
- REST API integration
- Performance metrics and logging
- Advanced feature summary

**Recommended for**: Learning all features, comprehensive overview

**Prerequisites**: 
- code-explainer package installed
- PyTorch and transformers
- Optional: Running API server for integration examples

**Usage**:
jupyter notebook code_explainer_comprehensive_guide.ipynb

### 2. code_explainer_api_quickstart.ipynb
Quick start guide for REST API usage:
- Server setup
- Single code explanation
- Batch processing
- Retrieval-augmented explanations
- Error handling
- Performance benchmarking
- Available models

**Recommended for**: REST API users, production integration

**Prerequisites**:
- API server running (see below)
- requests library
- Python 3.8+

**Usage**:
1. Start API server:
   uvicorn app:app --reload --host 0.0.0.0 --port 8000

2. Run notebook:
   jupyter notebook code_explainer_api_quickstart.ipynb

## Quick Start

### Installation
```bash
pip install code-explainer
```

### Python Library Usage
```python
from code_explainer import CodeExplainer

explainer = CodeExplainer(model_name="codet5-base")
explanation = explainer.explain_code("x = 42")
print(explanation)
```

### REST API Usage
```bash
# Start server
uvicorn app:app --reload

# Explain code via curl
curl -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{
    "code": "x = 42",
    "language": "python"
  }'
```

## Features Demonstrated

| Feature | Comprehensive | API QuickStart |
|---------|--------------|----------------|
| Basic Explanation | ✓ | ✓ |
| Batch Processing | ✓ | ✓ |
| Multi-Agent | ✓ | - |
| Retrieval-Augmented | ✓ | ✓ |
| Sanitization | ✓ | - |
| Caching | ✓ | - |
| Logging | ✓ | - |
| Performance Metrics | ✓ | ✓ |
| Error Handling | ✓ | ✓ |
| Model Selection | - | ✓ |

## Model Options

- **codet5-base**: General purpose code explanation
- **codet5-small**: Lightweight, faster inference
- **codebert-base**: Code embeddings
- **codellama-instruct**: Instruct-tuned variant
- **starcoder2-instruct**: Star Coder model

## Configuration

### Environment Variables
```bash
# Model selection
CODE_EXPLAINER_MODEL=codet5-base

# Device selection
CODE_EXPLAINER_DEVICE=cuda  # or mps, cpu

# Cache configuration
CODE_EXPLAINER_CACHE_EMBEDDING_TTL=3600
CODE_EXPLAINER_CACHE_EXPLANATION_TTL=7200

# API configuration
CODE_EXPLAINER_API_HOST=0.0.0.0
CODE_EXPLAINER_API_PORT=8000

# Data governance
CODE_EXPLAINER_DATA_RETENTION_DAYS=30
CODE_EXPLAINER_DATA_STORAGE_DISABLED=0
```

## Performance Tips

1. **Batch Processing**: Use batch API for multiple codes
2. **Caching**: Leverage LRU and TTL caches
3. **Quantization**: Use 8-bit quantization for memory constraints
4. **Device**: Use GPU (CUDA/MPS) for faster inference
5. **Strategy**: Choose retrieval-augmented for better quality

## Troubleshooting

### Out of Memory
```bash
# Use quantization
export CODE_EXPLAINER_PRECISION=8bit

# Or reduce batch size
codes = batch[:16]  # Process 16 at a time
```

### Slow Inference
```bash
# Use smaller model
explainer = CodeExplainer(model_name="codet5-small")

# Or enable caching
from code_explainer import get_model_cache_info
print(get_model_cache_info())
```

### API Connection Issues
```bash
# Check server is running
curl http://localhost:8000/api/v1/health

# Verify port is correct
netstat -an | grep 8000
```

## Advanced Usage

### Custom Strategies
```python
# Symbolic analysis
explanation = explainer.explain_code(code, strategy="symbolic")

# Retrieval-augmented
explanation = explainer.explain_code(code, strategy="retrieval-augmented")

# Multi-agent consensus
from code_explainer.multi_agent import MultiAgentOrchestrator
orchestrator = MultiAgentOrchestrator(num_agents=3)
result = orchestrator.explain_with_consensus(code, explainer.explain_code)
```

### Data Governance
```python
from code_explainer.data_governance import log_data_access, log_data_lineage

log_data_access("request-123", "STORE", "code_snippet")
log_data_lineage("model_training", ["train.json"], ["model-v1.safetensors"])
```

### Input Validation
```python
from code_explainer.input_sanitization import InputValidator

code, language = InputValidator.validate_code_request(code, language)
```

## License

Code Explainer examples are provided under the same license as the main project.
See LICENSE file for details.

## Support

- GitHub Issues: https://github.com/...
- Documentation: https://code-explainer.readthedocs.io
- Email: support@code-explainer.dev
"""
