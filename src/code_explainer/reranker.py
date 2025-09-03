"""Reranker utilities and Maximal Marginal Relevance (MMR).

This module provides a light wrapper around sentence-transformers CrossEncoder
and a simple MMR implementation. Designed to be optional: if sentence-transformers
is not installed, create_reranker() returns None and callers should degrade gracefully.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:  # optional dependency
    from sentence_transformers import CrossEncoder  # type: ignore

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # pragma: no cover - environment dependent
    CrossEncoder = None  # type: ignore
    HAS_SENTENCE_TRANSFORMERS = False


class CrossEncoderReranker:
    """Thin wrapper over a CrossEncoder for pair scoring (query, doc)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", **kwargs: Any):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers not available")
        self.model_name = model_name
        # allow passing device or other kwargs through
        self.model = CrossEncoder(model_name, **kwargs)  # type: ignore[misc]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c.get("content", "")) for c in candidates]
        scores = self.model.predict(pairs)  # type: ignore[operator]
        # Ensure numpy array for indexing
        scores = np.asarray(scores)
        ranked = []
        for c, s in zip(candidates, scores.tolist()):
            item = dict(c)
            item["rerank_score"] = float(s)
            ranked.append(item)
        if score_threshold is not None:
            ranked = [r for r in ranked if r["rerank_score"] >= score_threshold]
        ranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        if top_k is not None:
            ranked = ranked[: max(0, int(top_k))]
        return ranked


class MaximalMarginalRelevance:
    """Greedy MMR selection with cosine similarity."""

    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = float(lambda_param)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray | List[np.ndarray]) -> np.ndarray | float:
        a = np.asarray(a, dtype=float)
        if isinstance(b, list):
            b = np.asarray(b, dtype=float)
        else:
            b = np.asarray(b, dtype=float)
        if b.ndim == 1:
            # single vector
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            return float(np.dot(a, b) / denom)
        # many vectors (N, D)
        a_norm = np.linalg.norm(a) or 1.0
        b_norms = np.linalg.norm(b, axis=1)
        b_norms[b_norms == 0] = 1.0
        sims = (b @ a) / (a_norm * b_norms)
        return sims

    def select(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        n = min(len(candidate_embeddings), len(candidates))
        if n == 0:
            return []
        # Trim to minimum length to avoid mismatch issues
        cand_embs = candidate_embeddings[:n]
        cand_items = candidates[:n]

        # Compute query similarities
        q_sims = [float(self._cosine_similarity(query_embedding, emb)) for emb in cand_embs]
        selected: List[int] = []
        remaining = set(range(n))
        k = min(max(0, int(top_k)), n)
        while len(selected) < k and remaining:
            best_idx = None
            best_score = -1e9
            for i in list(remaining):
                if not selected:
                    div = 0.0
                else:
                    # Max similarity to any already selected item
                    div = max(
                        float(self._cosine_similarity(cand_embs[i], cand_embs[j])) for j in selected
                    )
                score = self.lambda_param * q_sims[i] - (1.0 - self.lambda_param) * div
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        # Build results
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(selected, start=1):
            item = dict(cand_items[idx])
            item["mmr_rank"] = rank
            item["query_similarity"] = q_sims[idx]
            results.append(item)
        return results


def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", **kwargs: Any) -> Optional[CrossEncoderReranker]:
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        return CrossEncoderReranker(model_name, **kwargs)
    except Exception:
        return None


def create_mmr(lambda_param: float = 0.5) -> MaximalMarginalRelevance:
    return MaximalMarginalRelevance(lambda_param=lambda_param)
