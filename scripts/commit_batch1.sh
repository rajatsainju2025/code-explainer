#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

git add README.md
git commit -m "docs(readme): add compatibility notes for recent shims"

git add src/code_explainer/__init__.py
git commit -m "pkg: export clear_model_cache and get_model_cache_info at package level"

git add src/code_explainer/device_manager.py src/code_explainer/utils/device.py
git commit -m "perf(device): make availability checks robust and use absolute imports for lazy device loader"

git add src/code_explainer/model/core.py
git commit -m "feat(explainer): add explain_code_with_symbolic convenience shim"

git add src/code_explainer/model/properties.py
git commit -m "test(explainer): return CPU device when mocks injected to ease testing"

git --no-pager log --oneline -n 5
