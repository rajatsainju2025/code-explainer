#!/usr/bin/env python
"""Validate simple dataset schema for code-explainer.

Usage:
  python scripts/validate_dataset.py data/train.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REQUIRED_KEYS = {"code", "explanation"}

def validate(path: Path) -> int:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"ERROR: failed to read {path}: {e}")
        return 2
    if not isinstance(data, list):
        print("ERROR: dataset must be a list of objects")
        return 2
    ok = True
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            print(f"ERROR[{i}]: record is not an object")
            ok = False
            continue
        missing = REQUIRED_KEYS - set(rec.keys())
        if missing:
            print(f"ERROR[{i}]: missing keys {sorted(missing)}")
            ok = False
    if ok:
        print(f"OK: {path} valid with {len(data)} records")
        return 0
    return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_dataset.py <path.json>")
        sys.exit(2)
    sys.exit(validate(Path(sys.argv[1])))
