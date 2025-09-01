# Evaluation Recipes

This page collects handy, copy-pasteable evaluation recipes you can adapt to your project.

## 1) Quick contamination check (JSONL)

```python
from code_explainer.evaluation import run_contamination_detection

report = run_contamination_detection(
    train_file="data/train.jsonl",
    test_file="data/test.jsonl",
    output_file="evaluation/contamination_report.json",
    methods=["exact", "ngram", "substring"],
    fields=["code", "explanation"],
)
print("Contamination rate:", f"{report.contamination_rate:.1%}")
```

## 2) Minimal robustness test

```python
from code_explainer.evaluation import run_robustness_tests

# Your model prediction function
predict = lambda ex: explainer.explain_code(ex.get("code", ""))

report = run_robustness_tests(
    examples=test_examples[:200],
    predict_func=predict,
    output_file="evaluation/robustness_report.json",
    test_types=["typo", "case", "whitespace"],
)
print("Overall robustness:", f"{report.overall_robustness_score:.3f}")
```

## 3) CLI one-liners

```bash
# contamination
code-explainer eval-contamination \
  --train-data data/train.jsonl \
  --test-data data/test.jsonl \
  --methods exact ngram substring

# robustness
code-explainer eval-robustness \
  --test-data data/test.jsonl \
  --model-path ./results
```
