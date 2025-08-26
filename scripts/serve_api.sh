#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CODE_EXPLAINER_MODEL_PATH=${CODE_EXPLAINER_MODEL_PATH:-"./results"}
export CODE_EXPLAINER_CONFIG_PATH=${CODE_EXPLAINER_CONFIG_PATH:-"configs/default.yaml"}
uvicorn code_explainer.api.server:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
