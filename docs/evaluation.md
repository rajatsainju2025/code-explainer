# Evaluation

We report BLEU, ROUGE-L, BERTScore, and CodeBLEU.

- BLEU: `sacrebleu`
- ROUGE-L: `rouge-score`
- BERTScore: `bert-score` (optional)
- CodeBLEU: optional; install with `pip install codebleu` (fallback to BLEU if unavailable)

CLI:

```bash
code-explainer eval -c configs/default.yaml --max-samples 10 --preds-out examples/eval_results/preds.jsonl
```

A/B compare strategies:

```bash
python scripts/ab_compare_strategies.py --config configs/default.yaml --max-samples 5 \
  --strategies vanilla ast_augmented retrieval_augmented
```
