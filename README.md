# 🐍 Code Explainer

<div align="center">

[![CI](https://github.com/rajatsainju2025/code-explainer/actions/workflows/quality-assurance.yml/badge.svg)](https://github.com/rajatsainju2025/code-explainer/actions/workflows/quality-assurance.yml)
[![Codecov](https://codecov.io/gh/rajatsainju2025/code-explainer/branch/main/graph/badge.svg)](https://codecov.io/gh/rajatsainju2025/code-explainer)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![Security](https://img.shields.io/badge/Security-Bandit%20%7C%20Safety-green)](https://github.com/rajatsainju2025/code-explainer/security)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajatsainju2025/code-explainer/blob/main/examples/colab_quickstart.ipynb)
[![Project Board](https://img.shields.io/badge/Project-Next%2010%20Days-blue)](https://github.com/rajatsainju2025/code-explainer)
[![Docs](https://img.shields.io/badge/Docs-Website-blue)](https://rajatsainju2025.github.io/code-explainer)
[![Nightly](https://img.shields.io/github/actions/workflow/status/rajatsainju2025/code-explainer/docs-and-eval-nightly.yml?label=nightly%20docs%2Feval)](https://github.com/rajatsainju2025/code-explainer/actions/workflows/docs-and-eval-nightly.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen)](.github/dependabot.yml)

**A state-of-the-art, production-ready LLM-powered system for generating human-readable explanations of Python code with enhanced retrieval, security, and monitoring capabilities.**

[🚀 Quick Start](#quick-start) • [📖 Documentation](docs/) • [� Tutorial](https://rajatsainju2025.github.io/code-explainer/tutorial/) • [�🔧 Installation](#installation) • [💡 Examples](#examples) • [🤝 Contributing](#contributing) • [💬 Discussions](https://github.com/rajatsainju2025/code-explainer/discussions)

</div>

---

## ✨ Features

Device portability and intelligent explanations:
- Unified DeviceManager selects the best device automatically (CUDA > MPS > CPU) with safe fallbacks
- Precision control via CODE_EXPLAINER_PRECISION (fp32, fp16, bf16, 8bit)
- Optional IntelligentExplanationGenerator for adaptive, audience-aware explanations

### 🧠 **Core AI Capabilities**
- **Advanced AI Models**: Fine-tuned CodeT5, CodeBERT, and GPT models for accurate explanations
- **Enhanced RAG**: Retrieval-Augmented Generation with FAISS, BM25, and hybrid search
- **Cross-Encoder Reranking**: Improved relevance with sentence-transformers rerankers
- **MMR Diversity**: Maximal Marginal Relevance for diverse code examples
- **Multi-Agent Analysis**: Collaborative explanations from specialized agents
- **Symbolic Analysis**: Property-based testing and complexity analysis

### 🎯 **Smart Analysis & Prompting**
- **Multiple Strategies**: vanilla, ast_augmented, retrieval_augmented, execution_trace, and enhanced_rag
- **Code Understanding**: Support for functions, classes, algorithms, and data structures
- **Complexity Analysis**: Automatic time/space complexity detection
- **Error Pattern Recognition**: Common bug identification and debugging suggestions

### 🌐 **Production-Ready Interfaces**
- **REST API**: FastAPI with Prometheus metrics, rate limiting, and health checks
- **Web UI**: Streamlit and Gradio interfaces for interactive exploration
- **CLI Tools**: Comprehensive command-line interface with rich output
- **Python SDK**: Direct integration for developers

### 🔒 **Security & Safety**
- **Code Redaction**: Automatic PII and credential detection and redaction
- **Security Validation**: AST-based dangerous pattern detection  
- **Safe Execution**: Sandboxed code execution with resource limits
- **Input Validation**: Comprehensive request validation and sanitization

### 📊 **Monitoring & Observability**
- **Prometheus Metrics**: API performance, error rates, and P95/P99 latencies
- **Grafana Dashboard**: Pre-built monitoring dashboards
- **Structured Logging**: JSON logging with request IDs and tracing
- **Health Checks**: Comprehensive service health monitoring

### 🧪 **Advanced Evaluation & Testing**
- **Traditional Metrics**: BLEU, ROUGE-L, BERTScore, CodeBLEU for quantitative assessment
- **LLM-as-a-Judge**: Multi-judge consensus evaluation with GPT-4 and Claude
- **Preference Learning**: Pairwise comparisons and Bradley-Terry ranking
- **Contamination Detection**: Comprehensive data leakage detection (exact, n-gram, semantic)
- **Robustness Testing**: Adversarial testing with 7 transformation types
- **Comprehensive CLI**: Full evaluation pipeline with detailed reporting

### 🔮 **Continuous Integration & Deployment**
- **Quality Assurance**: Automated testing with pytest, coverage, and type checking
- **Release Automation**: Automated releases with changelogs and semantic versioning
- **Pre-commit Hooks**: Code formatting, linting, and security checks
- **Multi-environment Testing**: Testing across Python 3.8, 3.9, 3.10, 3.11, 3.12

### 🎯 **Developer Experience**
- **mkdocs Documentation**: Comprehensive documentation site with examples
- **Development Containers**: VS Code devcontainer for instant setup
- **Makefile Automation**: Common tasks simplified with make commands
- **nbstripout**: Clean notebook commits without outputs

---

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install code-explainer

# Or install from source
git clone https://github.com/rajatsainju2025/code-explainer.git
cd code-explainer
make install

# Or use Docker
docker run -p 8000:8000 rajatsainju/code-explainer:latest
```

### Basic Usage

```python
from code_explainer import CodeExplainer

# Initialize the explainer
explainer = CodeExplainer(strategy="enhanced_rag")

# Explain some code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

explanation = explainer.explain(code)
print(explanation)
```

### Web Interface

```bash
# Start the FastAPI server
make serve

# Or use Streamlit
make streamlit

# Or use Gradio
make gradio
```

### CLI Usage

```bash
# Explain a file
code-explainer explain --file examples/fibonacci.py --strategy enhanced_rag

# Use different strategies
code-explainer explain --file mycode.py --strategy vanilla
code-explainer explain --file mycode.py --strategy ast_augmented
code-explainer explain --file mycode.py --strategy retrieval_augmented
code-explainer explain --file mycode.py --strategy execution_trace

# Run evaluations
code-explainer eval --dataset humaneval --model codet5-small

# Research-driven evaluation (new)
# End-to-end research evaluation (contamination, dynamic, multi-agent, adversarial)
python -m code_explainer.cli_evaluation \
  --model-path ./results \
  --model-id demo-model \
  --dynamic-rounds 3 \
  --adversarial-tests 10 \
  --enable-multi-agent --parallel

See docs/research_evaluation_system.md for details.
# Evaluate on a local JSONL (supports provenance + self-consistency)
code-explainer eval -c configs/default.yaml -t data/examples/tiny_eval.jsonl --self-consistency 3 -o out/preds.jsonl

# Notes:
# - JSONL supported: each line should have {"code", "explanation", optional "source_ids"}
# - Provenance metrics appear when source_ids/sources present
# - Self-consistency: pass --self-consistency N to sample N generations per item

# Check security
code-explainer security --file suspicious_code.py

# Run golden tests
code-explainer golden-test --dataset core

For a 15-minute walkthrough, see the Zero to Results tutorial: docs/tutorials/zero_to_results.md
```

---

## 📊 Performance & Benchmarks

| Metric | CodeT5-Small | CodeT5-Base | GPT-3.5-Turbo | Our Enhanced RAG |
|--------|--------------|-------------|---------------|------------------|
| BLEU-4 | 0.42 | 0.48 | 0.55 | **0.61** |
| ROUGE-L | 0.38 | 0.44 | 0.52 | **0.58** |
| BERTScore | 0.71 | 0.76 | 0.82 | **0.85** |
| CodeBLEU | 0.35 | 0.41 | 0.48 | **0.54** |
| Human Rating | 3.2/5 | 3.6/5 | 4.1/5 | **4.4/5** |

*Benchmarked on HumanEval and MBPP datasets with human evaluators.*

---

## 🧪 Advanced Evaluation Framework

Our evaluation system implements state-of-the-art assessment methods following open evaluation best practices:

### Traditional Metrics
```bash
# Comprehensive traditional metrics
code-explainer evaluate \
  --test-data test.jsonl \
  --predictions predictions.jsonl \
  --metrics bleu rouge bertscore codebleu
```

### LLM-as-a-Judge Evaluation
```bash
# Multi-judge consensus evaluation
code-explainer eval-llm-judge \
  --test-data test.jsonl \
  --predictions predictions.jsonl \
  --judges gpt-4 claude-3-sonnet \
  --criteria accuracy clarity completeness
```

### Contamination Detection
```bash
# Detect data leakage between train/test
code-explainer eval-contamination \
  --train-data train.jsonl \
  --test-data test.jsonl \
  --methods exact ngram substring semantic
```

### Robustness Testing
```bash
# Test model robustness under adversarial conditions
code-explainer eval-robustness \
  --test-data test.jsonl \
  --model-path ./results \
  --test-types typo case whitespace punctuation \
  --severity-levels 0.05 0.1 0.2
```

### Preference-Based Evaluation
```bash
# Compare models using pairwise preferences
code-explainer eval-preference \
  --test-data test.jsonl \
  --predictions-a model_a.jsonl \
  --predictions-b model_b.jsonl \
  --use-bradley-terry
```

**📖 See our [Advanced Evaluation Tutorial](docs/advanced_evaluation_tutorial.md) for comprehensive examples and best practices.**

---

## 🏗️ Architecture

```mermaid
graph TB
    A[Code Input] --> B[Security Validation]
    B --> C[AST Analysis]
    C --> D[Strategy Selection]
    
    D --> E1[Vanilla LLM]
    D --> E2[AST-Augmented]
    D --> E3[Enhanced RAG]
    D --> E4[Multi-Agent]
    
    E3 --> F[Vector Store]
    E3 --> G[BM25 Index]
    E3 --> H[Cross-Encoder Reranker]
    
    E1 --> I[Response Synthesis]
    E2 --> I
    E3 --> I
    E4 --> I
    
    I --> J[Quality Validation]
    J --> K[Security Redaction]
    K --> L[Final Explanation]
```

---

<!-- Removed duplicate architecture/performance section to avoid confusion -->

## 🔧 Configuration

The system is highly configurable through YAML files:

```yaml
# configs/custom.yaml
model:
  name: "microsoft/CodeGPT-small-py"
  max_length: 512
  temperature: 0.7

training:
  num_train_epochs: 100
  per_device_train_batch_size: 8
  learning_rate: 5e-5

prompt:
  template: "Explain this Python code:\n```python\n{code}\n```\nExplanation:"
```

## 📦 Model Presets

Use ready-made presets to switch models quickly:

| Preset | Arch | Base Model | Config | Train | Evaluate |
|-------|------|------------|--------|-------|----------|
| DistilGPT-2 (default) | causal | distilgpt2 | `configs/default.yaml` | `cx-train -c configs/default.yaml` | `code-explainer eval -c configs/default.yaml` |
| CodeT5 Small | seq2seq | Salesforce/codet5-small | `configs/codet5-small.yaml` | `cx-train -c configs/codet5-small.yaml` | `code-explainer eval -c configs/codet5-small.yaml` |
| CodeT5 Base | seq2seq | Salesforce/codet5-base | `configs/codet5-base.yaml` | `cx-train -c configs/codet5-base.yaml` | `code-explainer eval -c configs/codet5-base.yaml` |
| CodeGPT Small (CodeBERT family) | causal | microsoft/CodeGPT-small-py | `configs/codebert-base.yaml` | `cx-train -c configs/codebert-base.yaml` | `code-explainer eval -c configs/codebert-base.yaml` |
| StarCoderBase 1B | causal | bigcode/starcoderbase-1b | `configs/starcoderbase-1b.yaml` | `cx-train -c configs/starcoderbase-1b.yaml` | `code-explainer eval -c configs/starcoderbase-1b.yaml` |
| StarCoder2 Instruct | causal | bigcode/starcoder2-3b | `configs/starcoder2-instruct.yaml` | `cx-train -c configs/starcoder2-instruct.yaml` | `code-explainer eval -c configs/starcoder2-instruct.yaml` |
| CodeLlama Instruct | causal | codellama/CodeLlama-7b-Instruct-hf | `configs/codellama-instruct.yaml` | `cx-train -c configs/codellama-instruct.yaml` | `code-explainer eval -c configs/codellama-instruct.yaml` |

Data paths in each config default to the tiny examples in `data/`. Override any path via CLI flags (e.g., `--data` for training or `--test-file` for eval).

## 📖 Documentation

### Training Your Own Model

```python
from code_explainer import CodeExplainerTrainer

# Initialize trainer with custom config
trainer = CodeExplainerTrainer("configs/custom.yaml")

# Train on custom dataset
trainer.train(data_path="data/my_dataset.json")
```

### Advanced Usage

```python
# Batch processing
codes = ["print('hello')", "x = [1,2,3]", "def add(a,b): return a+b"]
explanations = explainer.explain_code_batch(codes)

# Prompt strategy (CLI)
# From API
# POST /explain {"code": "...", "strategy": "ast_augmented"}

# A/B compare strategies
python scripts/ab_compare_strategies.py --config configs/default.yaml --max-samples 5 \
  --strategies vanilla ast_augmented retrieval_augmented
```

## 🧩 Prompt Strategies

See `docs/strategies.md` for details on: vanilla | ast_augmented | retrieval_augmented | execution_trace, including safety notes and examples.

## 💡 Examples

See quick-start examples in `examples/` (training, evaluation, and serving with presets). Start here:

- examples/README.md
- examples/preset_switching.md
- examples/eval_report_template.md

Contribute examples/data: see the discussion “Call for community samples (tiny datasets)” in the Discussions tab.

<details>
<summary>📝 Example Explanations</summary>

**Input:**
```python
class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
        return self.balance
```

**Output:**
> This code defines a `BankAccount` class that represents a simple bank account. The `__init__` method initializes the account with an optional starting balance (defaulting to 0). The `deposit` method adds money to the account and returns the new balance.

</details>

## 🛠️ Development

```bash
git clone https://github.com/rajatsainju2025/code-explainer.git
cd code-explainer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

Additional tools:
- Makefile targets: install, format, lint, type, precommit, test, clean
- Devcontainer: `.devcontainer/devcontainer.json` for a ready-made VS Code container

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=code_explainer --cov-report=html

# Run specific test
pytest tests/test_model.py::test_explain_code
```

For scope, speed, and coverage goals, see the testing strategy discussion: `.github/DISCUSSIONS.md`.

Planning & Roadmap
- Plan review: docs/plan_review.md
- Roadmap: NEXT_PHASE_ROADMAP.md
- Reimagination: REIMAGINE.md

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

```dockerfile
# Build image
docker build -t code-explainer .

# Run web interface
# Run training
docker run -v $(pwd)/data:/app/data code-explainer train --data /app/data/train.json
```

## 📈 Roadmap

- [ ] **Multi-language Support**: JavaScript, Java, C++, etc.
- [ ] **Advanced Models**: Integration with CodeT5, CodeBERT, StarCoder
- [ ] **VS Code Extension**: Direct integration with development environment  
- [ ] **API Service**: RESTful API for integration with other tools
- [ ] **Performance Optimization**: Model quantization and optimization
- [ ] **Enterprise Features**: Authentication, usage tracking, custom deployments

## 📅 10-Day Contribution Plan

We are running a focused 10-day sprint targeting 10–15 meaningful contributions per day. See the detailed plan:

- CONTRIBUTION_PLAN_10_DAYS.md
- Track progress on the Project board (link will be added once created)
## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.
4. Push to the branch (`git push origin feature/amazing-feature`)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the amazing Transformers library
- [OpenAI](https://openai.com/) for GPT model architecture inspiration
- The open-source community for various tools and libraries

- **Author**: Rajat Sainju
- **Email**: your.email@example.com
- **GitHub**: [@rajatsainju2025](https://github.com/rajatsainju2025)
- **Project Link**: [https://github.com/rajatsainju2025/code-explainer](https://github.com/rajatsainju2025/code-explainer)

---

<div align="center">

</div>

## 💬 Join the community
- Start here: https://github.com/rajatsainju2025/code-explainer/discussions/4
- General Q&A and ideas: Discussions tab

## API (FastAPI)

Run the FastAPI server (example):

```bash
uvicorn code_explainer.api.server:app --host 0.0.0.0 --port 8000
```

Endpoints:
- GET /health → {"status": "ok"}
- GET /version → {"version": <semver>}
- GET /strategies → list of supported strategies
- POST /explain {code: str, strategy?: str} → {explanation: str}

More: see `docs/api.md` and `docs/strategies.md`.