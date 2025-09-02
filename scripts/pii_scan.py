#!/usr/bin/env python3
"""Lightweight PII scan for data/ changes.

Scans JSON/JSONL files in data/ for emails, SSNs, credit cards, and phone numbers.
Exits non-zero if matches are found.
"""
from __future__ import annotations
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b"), "CreditCard"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "Email"),
    (re.compile(r"\b\d{3}-\d{3}-\d{4}\b"), "Phone"),
]


def scan_text(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for rx, label in PII_PATTERNS:
        for m in rx.finditer(text):
            findings.append({
                "label": label,
                "match": m.group(0),
                "start": m.start(),
                "end": m.end(),
            })
    return findings


def scan_file(path: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, start=1):
                    f = scan_text(line)
                    for x in f:
                        x["line"] = i
                        x["file"] = str(path)
                    findings.extend(f)
        else:
            text = path.read_text(encoding="utf-8")
            findings = scan_text(text)
            for x in findings:
                x["file"] = str(path)
    except Exception as e:
        print(f"WARN: failed to scan {path}: {e}")
    return findings


def main() -> int:
    root = Path("data")
    if not root.exists():
        print("No data/ directory to scan.")
        return 0
    all_findings: List[Dict[str, Any]] = []
    for p in root.rglob("*.json*"):
        all_findings.extend(scan_file(p))
    if all_findings:
        print("PII scan found potential matches:")
        for f in all_findings[:100]:  # cap output
            print(f"- {f['file']}: {f.get('line','?')} {f['label']}: {f['match']}")
        print(f"Total findings: {len(all_findings)}")
        return 1
    print("PII scan clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
