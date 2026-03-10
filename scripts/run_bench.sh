#!/usr/bin/env bash
# Small helper to run available quick benchmarks
set -euo pipefail
python3 -m pip install -q -r requirements.txt || true
python3 scripts/bench_hash.py
python3 scripts/bench_regex.py
