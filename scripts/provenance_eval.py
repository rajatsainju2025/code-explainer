#!/usr/bin/env python3
"""Compute provenance metrics for explanations.

Input JSONL with fields:
  id, explanation, sources: ["id1", "id2", ...]
Outputs overall precision/recall averages.
"""
import argparse
import json
from pathlib import Path

from src.code_explainer.metrics.provenance import provenance_scores


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="JSONL with explanation and sources")
    ap.add_argument("--out", default="out/provenance.json")
    args = ap.parse_args()

    items = list(load_jsonl(Path(args.preds)))

    precs, recs = [], []
    for ex in items:
        s = provenance_scores(ex.get("explanation", ""), ex.get("sources", []))
        precs.append(s.precision)
        recs.append(s.recall)

    result = {
        "precision": sum(precs) / (len(precs) or 1),
        "recall": sum(recs) / (len(recs) or 1),
        "n": len(items),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
