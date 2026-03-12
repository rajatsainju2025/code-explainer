# Device Testing

This document explains how to run device-related tests locally and in CI.

Guidance:

- Prefer mocked tests (`tests/integration`) for CI to avoid hardware dependencies.
- When running device-specific unit tests, set `CODE_EXPLAINER_DEVICE` to `cpu`
  to avoid requiring GPU/MPS; set `CODE_EXPLAINER_PRECISION` to control precision.

Example:

```bash
export CODE_EXPLAINER_DEVICE=cpu
export CODE_EXPLAINER_PRECISION=fp32
pytest -q tests/unit/test_device_manager_fallback.py
```

CI: Add an integration job that runs the mocked integration tests and core unit
tests to detect regressions quickly.
