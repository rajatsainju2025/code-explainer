# Ablation Guide

Run controlled ablations across strategies, k, reranker, and temperature.

Example (stub script):
```bash
python scripts/ablation_matrix.py --config configs/default.yaml --strategies vanilla enhanced_rag --k 3 5 --temperature 0.0 0.7
```

Interpretation
- Plot metric deltas vs. parameters
- Prefer robust settings with minimal variance and cost
