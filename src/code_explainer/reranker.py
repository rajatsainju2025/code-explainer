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
        
        # Create pairs efficiently with list comprehension
        pairs = [(query, c.get("content", "")) for c in candidates]
        scores = self.model.predict(pairs)  # type: ignore[operator]
        scores = np.asarray(scores)
        
        # Create result list with scores in single pass
        ranked = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores.tolist())
        ]
        
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

        # Vectorized query similarities computation
        cand_embs_array = np.array(cand_embs)
        q_sims = self._cosine_similarity(query_embedding, cand_embs_array)
        if np.isscalar(q_sims):
            q_sims = np.array([q_sims])
        else:
            q_sims = np.asarray(q_sims)
        
        selected: List[int] = []
        remaining = set(range(n))
        k = min(max(0, int(top_k)), n)
        
        while len(selected) < k and remaining:
            remaining_list = list(remaining)
            if not selected:
                # First selection - just pick highest query similarity
                best_idx = remaining_list[np.argmax(q_sims[remaining_list])]
            else:
                # Vectorized diversity calculation
                selected_embs = cand_embs_array[selected]
                remaining_embs = cand_embs_array[remaining_list]
                
                # Compute similarities between remaining and selected items
                sim_matrix = self._cosine_similarity(remaining_embs, selected_embs)
                if isinstance(sim_matrix, (float, np.floating)):
                    sim_matrix = np.array([[sim_matrix]])
                elif isinstance(sim_matrix, np.ndarray) and sim_matrix.ndim == 1:
                    sim_matrix = sim_matrix.reshape(-1, 1)
                
                # Max similarity for each remaining item to any selected
                max_divs = np.max(sim_matrix, axis=1)
                
                # MMR scores
                relevance_scores = q_sims[remaining_list]
                mmr_scores = self.lambda_param * relevance_scores - (1.0 - self.lambda_param) * max_divs
                
                best_local_idx = np.argmax(mmr_scores)
                best_idx = remaining_list[best_local_idx]
            
            selected.append(best_idx)
            remaining.remove(best_idx)

        # Build results
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(selected, start=1):
            item = dict(cand_items[idx])
            item["mmr_rank"] = rank
            item["query_similarity"] = float(q_sims[idx])
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
