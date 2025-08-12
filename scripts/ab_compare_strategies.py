#!/usr/bin/env python
"""
Tiny A/B comparison of prompt strategies on data/test.json.
Usage:
  python scripts/ab_compare_strategies.py --config configs/default.yaml --model-path ./results --max-samples 10 \
      --strategies vanilla ast_augmented retrieval_augmented

Outputs a small table of BLEU/ROUGE/BERTScore/CodeBLEU per strategy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from code_explainer.model import CodeExplainer
from code_explainer.metrics.evaluate import (
    compute_bleu,
    compute_rouge_l,
    compute_codebert_score,
    compute_codebleu,
)


def run_strategy(cfg: str, model_path: str, test_file: str, strategy: str, max_samples: int | None) -> dict:
    explainer = CodeExplainer(model_path=model_path, config_path=cfg)
    with open(test_file, 'r') as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[: max(0, int(max_samples))]

    refs: List[str] = []
    preds: List[str] = []
    for ex in data:
        code = ex.get('code', '')
        ref = ex.get('explanation', '')
        pred = explainer.explain_code(code, strategy=strategy)
        refs.append(ref)
        preds.append(pred)

    return {
        'bleu': compute_bleu(refs, preds),
        'rougeL': compute_rouge_l(refs, preds),
        'bert': compute_codebert_score(refs, preds),
        'codebleu': compute_codebleu(refs, preds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', default='configs/default.yaml')
    p.add_argument('--model-path', '-m', default='./results')
    p.add_argument('--test-file', '-t', default=None)
    p.add_argument('--max-samples', type=int, default=10)
    p.add_argument('--strategies', nargs='+', default=['vanilla', 'ast_augmented'])
    args = p.parse_args()

    explainer = CodeExplainer(model_path=args.model_path, config_path=args.config)
    if args.test_file is None:
        test_file = explainer.config.get('data', {}).get('test_file', 'data/test.json')
    else:
        test_file = args.test_file

    results = {}
    for s in args.strategies:
        results[s] = run_strategy(args.config, args.model_path, test_file, s, args.max_samples)

    # Print simple table
    cols = ['Strategy', 'BLEU', 'ROUGE-L', 'BERT', 'CodeBLEU']
    print("\t".join(cols))
    for s, m in results.items():
        print(f"{s}\t{m['bleu']:.4f}\t{m['rougeL']:.4f}\t{m['bert']:.4f}\t{m['codebleu']:.4f}")


if __name__ == '__main__':
    main()
