"""Self-consistency and stability metrics.

Compute agreement among multiple generated explanations for the same input.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List

from .evaluate import compute_bleu, compute_rouge_l


@dataclass
class SelfConsistency:
    avg_pairwise_bleu: float
    avg_pairwise_rougeL: float
    n_samples: int


def pairwise_scores(texts: List[str]) -> SelfConsistency:
    pairs = list(itertools.combinations(texts, 2))
    if not pairs:
        return SelfConsistency(1.0, 1.0, len(texts))

    bleus: List[float] = []
    rouges: List[float] = []
    for a, b in pairs:
        bleus.append(compute_bleu([a], [b]))
        rouges.append(compute_rouge_l([a], [b]))

    return SelfConsistency(
        avg_pairwise_bleu=sum(bleus) / len(bleus),
        avg_pairwise_rougeL=sum(rouges) / len(rouges),
        n_samples=len(texts),
    )
