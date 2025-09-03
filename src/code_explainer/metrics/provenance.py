"""Provenance metrics for RAG-style explanations.

We assume explanations may include citations like [1], [2], or [snippet_id].
Given a set of retrieved sources with allowed IDs, compute:
- citation_precision: fraction of cited IDs that are actually in sources
- citation_recall: fraction of source IDs that are cited at least once
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple


CITATION_PATTERN = re.compile(r"\[(?P<cid>[^\]]+)\]")


def extract_citations(text: str) -> List[str]:
    return [m.group("cid").strip() for m in CITATION_PATTERN.finditer(text or "")]


@dataclass
class ProvenanceScores:
    precision: float
    recall: float
    f1: float
    hallucination_rate: float  # fraction of cited IDs that are not in sources
    cited_ids: List[str]
    true_ids: List[str]


def provenance_scores(explanation: str, source_ids: Iterable[str]) -> ProvenanceScores:
    citations: List[str] = extract_citations(explanation)
    cited: Set[str] = set(citations)
    sources: Set[str] = set(str(s) for s in source_ids)

    if not cited and not sources:
        return ProvenanceScores(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            hallucination_rate=0.0,
            cited_ids=[],
            true_ids=[],
        )

    tp = len(cited & sources)
    precision = tp / (len(cited) or 1)
    recall = tp / (len(sources) or 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    halluc = 0.0 if not citations else max(0.0, 1.0 - precision)
    return ProvenanceScores(
        precision=precision,
        recall=recall,
        f1=f1,
        hallucination_rate=halluc,
        cited_ids=citations,
        true_ids=sorted(sources),
    )


def highlight_citations(explanation: str, source_ids: Iterable[str]) -> Tuple[str, List[Tuple[str, bool]]]:
    """Return explanation with [id] citations annotated as valid/invalid.

    Returns a tuple of (annotated_markdown, list_of_(id, is_valid)).
    """
    sources = set(str(s) for s in source_ids)
    parts: List[str] = []
    validity: List[Tuple[str, bool]] = []
    last = 0
    for m in CITATION_PATTERN.finditer(explanation or ""):
        cid = m.group("cid").strip()
        is_valid = cid in sources if sources else False
        parts.append(explanation[last:m.start()])
        badge = f"[`{cid}` ✅]" if is_valid else f"[`{cid}` ❌]"
        parts.append(badge)
        validity.append((cid, is_valid))
        last = m.end()
    parts.append(explanation[last:])
    return ("".join(parts), validity)
