#!/usr/bin/env python3
"""Stub for running small ablation sweeps and writing a CSV summary."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

# Lightweight integration with CLI open-eval runner
try:
    from code_explainer.open_evals import run_eval as run_open_eval
except Exception:
    run_open_eval = None  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strategies", nargs="+", default=["vanilla", "enhanced_rag"])
    ap.add_argument("--k", nargs="+", type=int, default=[3])
    ap.add_argument("--temperature", nargs="+", type=float, default=[0.7])
    ap.add_argument("--out", default="out/ablation.csv")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "k", "temperature", "dataset", "accuracy"])  # minimal real metric
        for s in args.strategies:
            for k in args.k:
                for t in args.temperature:
                    dataset_id = "demo-addsub"
                    accuracy = 0.0
                    if run_open_eval:
                        try:
                            m = run_open_eval(dataset_id, out_csv=None, out_json=None)
                            accuracy = float(m.get("accuracy", 0.0))
                        except Exception:
                            accuracy = 0.0
                    w.writerow([s, k, t, dataset_id, accuracy])
    print(f"Wrote ablation CSV to {args.out}")


if __name__ == "__main__":
    main()
