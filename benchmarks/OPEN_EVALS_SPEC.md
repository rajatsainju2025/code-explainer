# Open Evals Spec

This spec defines how to add new evaluation datasets and tasks.

## Dataset Format

Use JSONL with fields:

- id: unique string
- code: code snippet (Python for now)
- reference: target explanation text
- metadata: {topic, difficulty, source}

## File Layout

```
benchmarks/
  datasets/
    humaneval.jsonl
    mbpp.jsonl
    security_redaction.jsonl
```

## Metrics

- bleu, rouge_l, bertscore, codebleu (optional)
- latency_p50, latency_p95, latency_p99
- failure_rate

## Submission Checklist

- Dataset source and license
- Script to generate or download
- Small sample subset for CI
