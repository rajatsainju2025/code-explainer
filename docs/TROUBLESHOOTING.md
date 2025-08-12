# Troubleshooting

Common issues and fixes.

- Missing metric packages
  - Error: `ModuleNotFoundError: rouge_score` or `bert_score`
  - Fix: `pip install -e .[dev]` or install individually: `pip install rouge-score bert-score`
- CodeBLEU not installed
  - Symptom: CodeBLEU shows 0.0 or falls back to BLEU
  - Fix (optional): `pip install codebleu`
- Large model OOM
  - Use 8-bit loading: set `model.load_in_8bit: true` and lower `max_length`
  - Try smaller preset config (e.g., `configs/codet5-small.yaml`)
- macOS MPS quirks
  - If hangs/errors: set `PYTORCH_ENABLE_MPS_FALLBACK=1` or use CPU (`export CUDA_VISIBLE_DEVICES=''`)
- FastAPI not starting
  - Ensure `uvicorn` installed; run `make api` or `scripts/run_api.sh`
- Pre-commit failures
  - Run `make precommit` to auto-fix; install hooks with `pre-commit install`
- Tests slow
  - Use `pytest -q -k test_prompt_strategies` or `make test` (tiny data)
