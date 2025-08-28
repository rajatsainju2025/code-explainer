# Evaluation

This guide explains how to run open evaluations and golden tests for Code Explainer.

## Open Evaluations

We support evaluating on a range of public datasets:

- HumanEval (text-based explanation targets)
- MBPP (Programming Problems)
- CodeContests-derived tasks
- Custom community datasets (JSONL)

### Run

```bash
# Evaluate a specific model on a dataset
code-explainer eval --dataset humaneval --model codet5-small --report out/humaneval_report.md

# Run against multiple models
code-explainer eval --dataset mbpp --model codet5-small codet5-base gpt-3.5-turbo \
  --report out/mbpp_compare.md --metrics bleu rouge bertscore codebleu

# Golden tests (regression)
code-explainer golden-test --dataset core --report out/golden.md
```

### Metrics

- BLEU-4, ROUGE-L, BERTScore, CodeBLEU
- Human rating support via CSV rubric
- Latency distribution: P50/P95/P99
- Failure rate and stability across reruns

### Reports

Outputs Markdown and JSON reports with detailed per-item results, aggregate metrics, and charts.

## Reproducibility

- Deterministic seeds for all runs
- Dataset version pinning
- Model hash/version recorded
- Environment capture (pip freeze, CUDA, OS)

Use `--save-artefacts out/run_YYYYMMDD_hhmm/` to store all logs, configs, and results.

## Custom Datasets

Provide a JSONL with fields:

```json
{"id":"item-1","code":"def add(a,b): return a+b","reference":"Adds two numbers"}
```

Run:

```bash
code-explainer eval --dataset path/to/data.jsonl --model codet5-small
```
