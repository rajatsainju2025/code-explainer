# Advanced Tutorial: LLM-as-a-Judge vs Traditional Metrics

This tutorial shows how to run both traditional metrics (BLEU/ROUGE/CodeBLEU) and LLM-as-a-Judge evaluations, and how to interpret disagreements.

Steps
1) Produce predictions JSONL using the CLI eval with `-o`:
```bash
code-explainer eval -c configs/default.yaml -t data/examples/tiny_eval.jsonl -o out/preds.jsonl --max-samples 5
```

2) Run LLM Judge on the same data (requires API keys):
```bash
code-explainer eval-llm-judge \
  --test-data data/examples/tiny_eval.jsonl \
  --predictions out/preds.jsonl \
  --judges gpt-4 claude-3-sonnet \
  --criteria accuracy clarity completeness \
  --output out/llm_judge_report.json
```

3) Compare results and investigate disagreements:
- Where BLEU/ROUGE are high but judge scores are low: check hallucinations or poor clarity.
- Where judge scores are high but BLEU is low: check paraphrasing and synonyms.
- Use `scripts/provenance_card.py` to generate per-example cards for qualitative review.

Notes
- Enable randomization for pairwise comparisons when using preference evals.
- Cache judge calls to control cost.