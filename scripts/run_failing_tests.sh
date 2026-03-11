#!/usr/bin/env bash
# Run a focused subset of tests that were failing recently
set -euo pipefail
pytest tests/test_symbolic.py::TestSymbolicIntegration::test_symbolic_explanation_integration -q
pytest tests/unit/test_device_manager.py::TestDeviceManager::test_get_optimal_device_mps_fallback -q
pytest tests/test_utils.py::test_get_device_returns_string -q
