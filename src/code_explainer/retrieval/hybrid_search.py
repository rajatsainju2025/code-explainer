"""Hybrid search combining multiple retrieval methods."""

from typing import Dict, List, Tuple

import numpy as np

from .bm25_index import BM25Index
from .faiss_index import FAISSIndex


class HybridSearch:
    """Combines FAISS and BM25 search results."""

    def __init__(self, faiss_index: FAISSIndex, bm25_index: BM25Index, alpha: float = 0.5):
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.alpha = alpha  # Weight for FAISS vs BM25

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform hybrid search combining FAISS and BM25."""
        faiss_scores: Dict[int, float] = {}
        bm25_scores: Dict[int, float] = {}

        # FAISS search
        try:
            distances, indices = self.faiss_index.search(query, k)
            for d, i in zip(distances[0], indices[0]):
                sim = 1.0 / (1.0 + float(d))  # Convert distance to similarity
                faiss_scores[int(i)] = sim
        except Exception:
            pass  # FAISS search failed

        # BM25 search
        try:
            scores, indices = self.bm25_index.search(query, k)
            for s, i in zip(scores, indices):
                bm25_scores[int(i)] = float(s)
        except Exception:
            pass  # BM25 search failed

        # Combine scores
        if not faiss_scores and not bm25_scores:
            return []

        # Normalize BM25 scores
        if bm25_scores:
            bm_min = min(bm25_scores.values())
            bm_max = max(bm25_scores.values())
            bm_range = (bm_max - bm_min) or 1.0
            bm25_norm = {i: (s - bm_min) / bm_range for i, s in bm25_scores.items()}
        else:
            bm25_norm = {}

        # Fuse scores
        candidates = set(faiss_scores) | set(bm25_norm)
        fused: List[Tuple[int, float]] = []
        for i in candidates:
            s_f = faiss_scores.get(i, 0.0)
            s_b = bm25_norm.get(i, 0.0)
            fused.append((i, self.alpha * s_f + (1 - self.alpha) * s_b))

        # Sort by fused score
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k]