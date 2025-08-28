#!/usr/bin/env python3
"""Run a quick smoke evaluation on a tiny dataset to validate the pipeline."""

import argparse
import json
from pathlib import Path

from src.code_explainer.model import CodeExplainer
from src.code_explainer.metrics.evaluate import (
    compute_bleu,
    compute_rouge_l,
    compute_codebert_score,
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="benchmarks/datasets/smoke.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--report", default="out/smoke_report.json")
    args = ap.parse_args()

    data = list(load_jsonl(Path(args.dataset)))
    explainer = CodeExplainer(config_path=args.config)

    refs, preds = [], []
    for ex in data:
        code = ex.get("code", "")
        ref = ex.get("reference", "")
        try:
            pred = explainer.explain_code(code, strategy="enhanced_rag")
        except Exception:
            pred = ""
        refs.append(ref)
        preds.append(pred)

    report = {
        "bleu": compute_bleu(refs, preds),
        "rougeL": compute_rouge_l(refs, preds),
        "bert": compute_codebert_score(refs, preds),
        "n": len(refs),
    }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
