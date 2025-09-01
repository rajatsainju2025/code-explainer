#!/usr/bin/env python3
"""Stub for running small ablation sweeps and writing a CSV summary."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


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
        w.writerow(["strategy", "k", "temperature", "bleu", "rougeL", "codebleu"])  # placeholder
        for s in args.strategies:
            for k in args.k:
                for t in args.temperature:
                    # TODO: integrate with CLI eval to get real numbers
                    w.writerow([s, k, t, 0.0, 0.0, 0.0])
    print(f"Wrote ablation CSV to {args.out}")


if __name__ == "__main__":
    main()
