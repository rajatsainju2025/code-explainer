"""Open evaluations module: datasets registry and runner.

This is a minimal scaffold to run standardized evaluations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Optional

from .model import CodeExplainer


@dataclass
class EvalSample:
    code: str
    reference: str


class Dataset:
    name: str
    license: str
    def load(self) -> List[EvalSample]:
        raise NotImplementedError


class DemoAddSub(Dataset):
    name = "demo-addsub"
    license = "MIT"
    def load(self) -> List[EvalSample]:
        data = [
            {"code": "def add(a,b): return a+b", "reference": "Adds two numbers."},
            {"code": "def sub(a,b): return a-b", "reference": "Subtracts two numbers."},
        ]
        return [EvalSample(**x) for x in data]


DATASETS: Dict[str, Dataset] = {
    DemoAddSub.name: DemoAddSub(),
}


def run_eval(dataset: str, model_path: str = "./results", config_path: str = "configs/default.yaml",
             out_csv: Optional[str] = None, out_json: Optional[str] = None) -> Dict[str, float]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    ds = DATASETS[dataset].load()
    explainer = CodeExplainer(model_path=model_path, config_path=config_path)

    refs: List[str] = []
    preds: List[str] = []
    for s in ds:
        refs.append(s.reference)
        try:
            preds.append(explainer.explain_code(s.code))
        except Exception:
            preds.append("")

    from .metrics.evaluate import compute_bleu, compute_rouge_l, compute_codebert_score

    metrics = {
        "bleu": float(compute_bleu(refs, preds)),
        "rougeL": float(compute_rouge_l(refs, preds)),
        "bert_score": float(compute_codebert_score(refs, preds)),
    }

    if out_csv:
        import csv
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as wf:
            w = csv.writer(wf)
            w.writerow(["code", "reference", "prediction"])
            for s, pred in zip(ds, preds):
                w.writerow([s.code, s.reference, pred])
    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as wf:
            json.dump({"refs": refs, "preds": preds, "metrics": metrics}, wf)

    return metrics