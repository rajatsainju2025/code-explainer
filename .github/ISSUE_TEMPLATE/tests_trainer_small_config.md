---
name: "Test: Trainer small config"
about: Add tests for trainer with a minimal tiny model config
labels: [good first issue, testing]
assignees: ''
---

### Summary
Add a unit test to run the trainer with a very small config using a tiny model, ensuring that model loading, dataset preprocessing, and trainer setup succeed.

### Acceptance Criteria
- A new test file `tests/test_trainer_small_config.py` that:
  - Uses `sshleifer/tiny-gpt2`
  - Builds a minimal JSON config
  - Calls `CodeExplainerTrainer.load_model()`, `load_dataset(None)`, `preprocess_dataset()`, and `setup_trainer()`
- Test should run quickly (< 60s) and not perform full training.

### References
- Existing tests in `tests/test_trainer.py`
- Tiny models used in CI
