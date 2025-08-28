"""Simple contamination detection utilities.

Detect potential train/eval overlap using token n-gram Jaccard similarity.
Not a guarantee, but a heuristic to flag suspicious overlaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple


def _tokenize(text: str) -> List[str]:
    return [t for t in text.replace("\r", "").split() if t]


def _ngrams(tokens: List[str], n: int = 5) -> Set[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


@dataclass
class OverlapResult:
    pair: Tuple[str, str]
    jaccard: float


def detect_overlap(
    eval_texts: Iterable[Tuple[str, str]],
    train_texts: Iterable[Tuple[str, str]],
    ngram: int = 5,
    threshold: float = 0.8,
) -> List[OverlapResult]:
    """Detect high-overlap pairs.

    eval_texts/train_texts: iterable of (id, text)
    Returns results where Jaccard >= threshold.
    """
    eval_grams = [(eid, _ngrams(_tokenize(txt), ngram)) for eid, txt in eval_texts]
    train_grams = [(tid, _ngrams(_tokenize(txt), ngram)) for tid, txt in train_texts]

    flagged: List[OverlapResult] = []
    for eid, eg in eval_grams:
        for tid, tg in train_grams:
            sim = jaccard(eg, tg)
            if sim >= threshold:
                flagged.append(OverlapResult((eid, tid), sim))
    flagged.sort(key=lambda r: r.jaccard, reverse=True)
    return flagged


def read_jsonl(path: Path) -> List[Tuple[str, str]]:
    import json

    out: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            _id = obj.get("id") or obj.get("name") or str(len(out))
            text = obj.get("code") or obj.get("text") or ""
            out.append((_id, text))
    return out
