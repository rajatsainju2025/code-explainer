---
name: "Type hints: model & trainer"
about: Add missing type hints and Optional guards in model.py and trainer.py
labels: [good first issue, enhancement]
assignees: ''
---

### Summary
Improve typing in `src/code_explainer/model.py` and `src/code_explainer/trainer.py`. Add explicit types for tokenizer/model, Optional checks, and avoid Any where possible.

### Acceptance Criteria
- Add type annotations to public methods and key private helpers.
- Ensure mypy passes for src/ with current config.
- No behavior changes; typing-only PR.

### References
- mypy config in `.mypy.ini`
- Existing TODOs in code comments (if any)
