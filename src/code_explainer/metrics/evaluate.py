"""Evaluation metrics for code explanations."""
from __future__ import annotations

from typing import Dict, List

from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score
except Exception:  # pragma: no cover
    bert_score = None  # type: ignore


def compute_bleu(references: List[str], predictions: List[str]) -> float:
    # Lightweight BLEU using sacrebleu if available
    try:
        import sacrebleu  # type: ignore

        return float(sacrebleu.corpus_bleu(predictions, [references]).score)
    except Exception:  # pragma: no cover
        return 0.0


def compute_rouge_l(references: List[str], predictions: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(references, predictions)]
    return float(sum(scores) / max(len(scores), 1))


def compute_codebert_score(references: List[str], predictions: List[str]) -> float:
    if bert_score is None:  # pragma: no cover
        return 0.0
    P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True)
    return float(F1.mean().item())
