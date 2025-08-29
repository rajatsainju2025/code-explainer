# Tutorial: Build a Robust Code Explainer with Enhanced RAG

This hands-on tutorial walks you through using Code Explainer end-to-end: install, retrieve, explain, evaluate, and monitor.

## 1. Install and Verify

```bash
# Create venv (recommended)
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Verify
code-explainer --help
```

## 2. Start the API and Web UIs

```bash
# FastAPI
uvicorn src.code_explainer.api.server:app --reload
# Streamlit
streamlit run streamlit_app.py
# Gradio
python src/code_explainer/web/gradio_app.py
```

Open http://localhost:8000/docs to explore REST endpoints.

## 3. Build Retrieval and Explain

```bash
# Optional: build an index for your repo (example paths)
code-explainer build-index --config configs/default.yaml --output-path data/code_retrieval_index.faiss

# Explain a file with enhanced RAG
code-explainer explain --file examples/fibonacci.py --strategy enhanced_rag
```

## 4. Security Redaction and Safety

```bash
# Scan a file or directory
code-explainer security --file suspicious.py
```

## 5. Evaluate (Open Evals + Golden Tests)

```bash
# Quick smoke eval
python scripts/run_smoke_eval.py --dataset benchmarks/datasets/smoke.jsonl

# HumanEval/MBPP (subset)
code-explainer eval --dataset humaneval --max-samples 10 --report out/humaneval.md

# Golden tests
code-explainer golden-test --dataset core --report out/golden.md
```
# JSONL and provenance/self-consistency supported
code-explainer eval -c configs/default.yaml -t data/examples/tiny_eval.jsonl --self-consistency 2 --max-samples 2
```

## 6. Advanced Evaluations

```bash
# Detect contamination
code-explainer detect-contamination \
  --eval-jsonl benchmarks/datasets/smoke.jsonl \
  --train-jsonl benchmarks/datasets/smoke.jsonl

# Self-consistency
code-explainer self-consistency --num-samples 5 --strategy enhanced_rag "def add(a,b): return a+b"

# Provenance metrics (citation precision/recall)
python scripts/provenance_eval.py --preds examples/provenance_samples.jsonl
```

## 7. Monitor and Observe

- Prometheus metrics on /metrics (FastAPI)
- Use `monitoring/grafana-dashboard.json` in Grafana

## 8. Reproducibility Checklist

- Pin configs in `configs/`
- Save artefacts `--save-artefacts out/run_.../`
- Capture env: `pip freeze` into run folder

## 9. Next Steps

- Explore docs: strategies, retrieval-advanced, evaluation, governance
- Open an RFC for a new eval task or strategy
