# Zero to Results in 15 Minutes

Get a working evaluation with cached small models and sample data.

Prereqs
- Python 3.10+
- macOS/Linux, 4GB RAM

Steps
1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2) Quick smoke eval (5 examples)
```bash
python scripts/run_smoke_eval.py --dataset benchmarks/datasets/smoke.jsonl
```

3) CLI eval with JSONL and self-consistency
```bash
code-explainer eval -c configs/default.yaml -t data/examples/tiny_eval.jsonl --self-consistency 2 --max-samples 2
```

4) Provenance preview (optional)
```bash
python scripts/provenance_card.py --preds examples/provenance_samples.jsonl --out out/cards/
```

5) Explore docs and next steps
- Read Evaluation and Advanced Evaluation Tutorial
- Try strategies: vanilla, ast_augmented, retrieval_augmented, enhanced_rag

Tips
- Use --max-samples in CI
- Cache explanations to speed up local loops
