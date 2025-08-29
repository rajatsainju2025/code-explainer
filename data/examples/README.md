# Data Examples

This folder contains tiny example datasets for quick evaluation and CI smoke tests.

- `tiny_eval.jsonl` — JSON Lines with fields per item:
  - `id` (str): unique identifier
  - `code` (str): the code snippet to explain
  - `explanation` (str): reference explanation
  - `source_ids` (list[str], optional): allowed citation IDs for provenance metrics

Run a quick eval with self-consistency and provenance:

```
code-explainer eval -c configs/default.yaml -t data/examples/tiny_eval.jsonl --self-consistency 2
```
