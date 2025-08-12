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


def compute_codebleu(references: List[str], predictions: List[str]) -> float:
    """Compute CodeBLEU if package available; else fallback to BLEU.
    CodeBLEU considers weighted n-gram, syntax, and semantic matches.
    """
    try:
        # Try lightweight import path if available
        from codebleu import calc_code_bleu  # type: ignore

        # Assume all Python for now; future: detect per-sample
        lang = "python"
        score_sum = 0.0
        n = max(1, len(references))
        for ref, pred in zip(references, predictions):
            try:
                scores = calc_code_bleu([ref], [pred], lang)
                score_sum += float(scores["codebleu"])
            except Exception:
                pass
        return float(score_sum / n)
    except Exception:
        # Fallback to BLEU if CodeBLEU not available
        return compute_bleu(references, predictions)
