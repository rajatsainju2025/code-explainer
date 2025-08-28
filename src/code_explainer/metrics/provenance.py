"""Provenance metrics for RAG-style explanations.

We assume explanations may include citations like [1], [2], or [snippet_id].
Given a set of retrieved sources with allowed IDs, compute:
- citation_precision: fraction of cited IDs that are actually in sources
- citation_recall: fraction of source IDs that are cited at least once
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Set


CITATION_PATTERN = re.compile(r"\[(?P<cid>[^\]]+)\]")


def extract_citations(text: str) -> List[str]:
    return [m.group("cid").strip() for m in CITATION_PATTERN.finditer(text or "")]


@dataclass
class ProvenanceScores:
    precision: float
    recall: float


def provenance_scores(explanation: str, source_ids: Iterable[str]) -> ProvenanceScores:
    citations: List[str] = extract_citations(explanation)
    cited: Set[str] = set(citations)
    sources: Set[str] = set(str(s) for s in source_ids)

    if not cited and not sources:
        return ProvenanceScores(precision=1.0, recall=1.0)

    tp = len(cited & sources)
    precision = tp / (len(cited) or 1)
    recall = tp / (len(sources) or 1)
    return ProvenanceScores(precision=precision, recall=recall)
