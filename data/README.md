# Dataset Schema and Guidelines

Files: train.json, eval.json, test.json

Schema (JSON list of objects):
- code: string (required)
- explanation: string (required)

Guidelines:
- Keep examples short and diverse (loops, functions, classes, I/O, comprehensions, errors, etc.)
- Use Python primarily for now; keep consistent style
- Avoid very long snippets (>30 lines) in tiny datasets
- Prefer deterministic, well-understood explanations

Validation:
- Run `python scripts/validate_dataset.py data/<file>.json`
- CI validates all three files
