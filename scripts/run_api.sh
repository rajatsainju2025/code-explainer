#!/usr/bin/env bash
set -euo pipefail
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
exec uvicorn code_explainer.api.server:app --host "$HOST" --port "$PORT" --reload
