#!/usr/bin/env python3
"""
Generate simple per-example provenance cards from predictions JSON/JSONL.

Input schema per line/object:
{
  "code": str,
  "prediction": str,
  "reference": str (optional),
  "source_ids": ["doc1", "doc2", ...] (optional)
}

Outputs one Markdown file per example with basic placeholders.
This is a minimal stub to be expanded with highlighting later.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

# local import
try:
    from code_explainer.metrics.provenance import provenance_scores, highlight_citations
except Exception:
    provenance_scores = None
    highlight_citations = None


def _read_any(p: Path):
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(p.read_text(encoding="utf-8"))


ess = """
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Predictions file (JSON/JSONL)")
    ap.add_argument("--out", required=True, help="Output directory for cards")
    args = ap.parse_args()

    preds_path = Path(args.preds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = _read_any(preds_path)
    for i, ex in enumerate(examples):
        code = ex.get("code", "")
        pred = ex.get("prediction", "") or ex.get("pred", "")
        ref = ex.get("reference", "")
        srcs: List[str] = ex.get("source_ids") or ex.get("sources") or []
        prov_md = []
        if provenance_scores and highlight_citations and (pred or ref):
            annotated, validity = highlight_citations(pred or ref, srcs)
            scores = provenance_scores(pred or ref, srcs)
            prov_md.extend([
                "## Citations",
                annotated,
                "",
                f"- Precision: {scores.precision:.3f}",
                f"- Recall: {scores.recall:.3f}",
                f"- F1: {scores.f1:.3f}",
                f"- Hallucination rate: {scores.hallucination_rate:.3f}",
            ])
        else:
            prov_md.extend(["## Citations", "(provenance metrics not available)"])
        md = [
            f"# Provenance Card #{i+1}",
            "",
            "## Code",
            "```python",
            code,
            "```",
            "",
            "## Prediction",
            pred or "",
            "",
            "## Reference (optional)",
            ref or "",
            "",
            "## Source IDs (optional)",
            ", ".join(map(str, srcs)) if srcs else "(none)",
            "",
            *prov_md,
            "",
            "> Note: Future versions will include cited span highlighting and coverage metrics.",
        ]
        (out_dir / f"card_{i+1:04d}.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {len(examples)} provenance cards to {out_dir}")


if __name__ == "__main__":
    main()
