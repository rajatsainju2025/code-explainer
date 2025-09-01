#!/usr/bin/env python3
"""Validate dataset intake YAML files in data/ against required fields."""
from __future__ import annotations
import sys
from pathlib import Path
import yaml

REQUIRED = {"id", "title", "description", "source_url", "license", "added_by", "added_at"}


def validate_file(path: Path) -> list[str]:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"{path}: YAML parse error: {e}"]
    if not isinstance(obj, dict):
        return [f"{path}: expected a mapping at top-level"]
    missing = REQUIRED - obj.keys()
    errs = []
    if missing:
        errs.append(f"{path}: missing fields: {', '.join(sorted(missing))}")
    return errs


def main(paths: list[str]) -> int:
    errs: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for y in path.glob("*.yaml"):
                errs.extend(validate_file(y))
        else:
            errs.extend(validate_file(path))
    if errs:
        print("\n".join(errs))
        return 1
    print("All intake files valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["data"]))
