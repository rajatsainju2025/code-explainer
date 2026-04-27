"""Hybrid search combining multiple retrieval methods with advanced features."""

import heapq
import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np

from .bm25_index import BM25Index
from .faiss_index import FAISSIndex

logger = logging.getLogger(__name__)

_SMALL_RESULTSET_THRESHOLD = 64


class FusionStrategy(Enum):
    """Strategies for fusing multiple retrieval scores."""
    LINEAR = "linear"  # Simple weighted linear combination
    RRF = "rrf"  # Reciprocal Rank Fusion
    DISTRIBUTION_BASED = "distribution_based"  # Distribution-based fusion


# Module-level dispatch avoids a per-call if/elif chain.
# Values are *method names* (strings) so no forward-reference issues.
_STRATEGY_METHOD: dict = {
    FusionStrategy.LINEAR: "_linear_fusion",
    FusionStrategy.RRF: "_rrf_fusion",
    FusionStrategy.DISTRIBUTION_BASED: "_distribution_fusion",
}


class AdvancedHybridSearch:
    """Advanced hybrid search with multiple fusion strategies and query expansion."""
    
    __slots__ = ('faiss_index', 'bm25_index', 'fusion_strategy', 'alpha', 'rrf_k')

    def __init__(self,
                 faiss_index: Optional[FAISSIndex] = None,
                 bm25_index: Optional[BM25Index] = None,
                 fusion_strategy: FusionStrategy = FusionStrategy.LINEAR,
                 alpha: float = 0.5,
                 rrf_k: int = 60):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got {rrf_k!r}")
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(self,
               query: str,
               k: int) -> List[Tuple[int, float]]:
        """Perform hybrid search combining FAISS and BM25 results."""
        faiss_results = self._faiss_search(query, k)
        bm25_results = self._bm25_search(query, k)

        if not faiss_results and not bm25_results:
            return []

        method_name = _STRATEGY_METHOD.get(self.fusion_strategy, "_linear_fusion")
        return getattr(self, method_name)(faiss_results, bm25_results, k)

    def _faiss_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform FAISS search."""
        if not self.faiss_index:
            return []

        try:
            distances, indices = self.faiss_index.search(query, k)
            # IndexFlatIP returns inner-product scores for L2-normalised vectors,
            # which are cosine similarities in [-1, 1].  Use them directly as
            # higher-is-better similarity scores; no further transformation needed.
            results = [(int(i), float(d)) for d, i in zip(distances[0], indices[0])]
            return results
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
            return []

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform BM25 search."""
        if not self.bm25_index:
            return []

        try:
            scores, indices = self.bm25_index.search(query, k)
            return [(int(i), float(s)) for s, i in zip(scores, indices)]
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            return []

    def _linear_fusion(self,
                      faiss_results: List[Tuple[int, float]],
                      bm25_results: List[Tuple[int, float]],
                      k: int) -> List[Tuple[int, float]]:
        """Linear fusion of FAISS and BM25 results with optimized numpy operations."""
        if not faiss_results and not bm25_results:
            return []
        
        # Convert to dictionaries for easy merging
        faiss_dict = dict(faiss_results)
        bm25_dict = dict(bm25_results)
        # dict_keys supports set subtraction in O(len(bm25_dict)) without
        # building an intermediate list.
        candidate_count = len(faiss_dict) + len(bm25_dict.keys() - faiss_dict.keys())

        if candidate_count <= _SMALL_RESULTSET_THRESHOLD:
            return self._linear_fusion_small(faiss_dict, bm25_dict, k)

        # Vectorized BM25 score normalization using numpy
        if bm25_dict:
            bm_scores = np.array(list(bm25_dict.values()))
            bm_min, bm_max = bm_scores.min(), bm_scores.max()
            bm_range = bm_max - bm_min if bm_max > bm_min else 1.0
            # Vectorized normalization
            normalized_scores = (bm_scores - bm_min) / bm_range
            bm25_dict = dict(zip(bm25_dict.keys(), normalized_scores))

        # Fuse scores with vectorized operations
        all_indices = set(faiss_dict.keys()) | set(bm25_dict.keys())
        indices_array = np.array(list(all_indices))
        
        faiss_scores = np.array([faiss_dict.get(idx, 0.0) for idx in indices_array])
        bm25_scores = np.array([bm25_dict.get(idx, 0.0) for idx in indices_array])
        
        # Vectorized fusion
        fused_scores = self.alpha * faiss_scores + (1 - self.alpha) * bm25_scores
        
        # Sort by fused score and return top k
        sorted_indices = np.argsort(-fused_scores)[:k]
        return [(int(indices_array[idx]), float(fused_scores[idx])) for idx in sorted_indices]

    def _linear_fusion_small(
        self,
        faiss_dict: Dict[int, float],
        bm25_dict: Dict[int, float],
        k: int,
    ) -> List[Tuple[int, float]]:
        """Low-overhead linear fusion for typical small top-k result sets."""
        normalized_bm25 = bm25_dict
        if bm25_dict:
            bm_min = min(bm25_dict.values())
            bm_max = max(bm25_dict.values())
            bm_range = (bm_max - bm_min) or 1.0
            normalized_bm25 = {
                idx: (score - bm_min) / bm_range
                for idx, score in bm25_dict.items()
            }

        fused_scores = {
            idx: self.alpha * faiss_dict.get(idx, 0.0)
            + (1 - self.alpha) * normalized_bm25.get(idx, 0.0)
            for idx in faiss_dict.keys() | normalized_bm25.keys()
        }
        return heapq.nlargest(k, fused_scores.items(), key=lambda item: item[1])

    def _rrf_fusion(self,
                   faiss_results: List[Tuple[int, float]],
                   bm25_results: List[Tuple[int, float]],
                   k: int) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion with vectorized operations."""
        rrf_scores: Dict[int, float] = {}

        # Pure Python RRF - faster than numpy for typical k=5-20
        if faiss_results:
            for rank, (idx, _) in enumerate(faiss_results, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        if bm25_results:
            for rank, (idx, _) in enumerate(bm25_results, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        # heapq.nlargest is O(n log k) vs O(n log n) for a full sort —
        # measurably faster when k << n (the typical case).
        return heapq.nlargest(k, rrf_scores.items(), key=lambda x: x[1])

    def _distribution_fusion(self,
                           faiss_results: List[Tuple[int, float]],
                           bm25_results: List[Tuple[int, float]],
                           k: int) -> List[Tuple[int, float]]:
        """Distribution-based fusion using score distributions with numpy optimization."""
        if len(faiss_results) + len(bm25_results) <= _SMALL_RESULTSET_THRESHOLD:
            return self._distribution_fusion_small(faiss_results, bm25_results, k)

        # Use numpy arrays for efficient computation
        faiss_scores = np.array([score for _, score in faiss_results]) if faiss_results else np.array([])
        bm25_scores = np.array([score for _, score in bm25_results]) if bm25_results else np.array([])

        # Calculate distribution statistics using numpy
        faiss_stats = self._calculate_distribution_stats(faiss_scores)
        bm25_stats = self._calculate_distribution_stats(bm25_scores)

        # Fuse using distribution-aware weighting with vectorized operations
        faiss_dict = dict(faiss_results)
        bm25_dict = dict(bm25_results)
        all_indices = set(faiss_dict.keys()) | set(bm25_dict.keys())
        indices_array = np.array(list(all_indices))
        
        faiss_scores_array = np.array([faiss_dict.get(idx, faiss_stats['mean']) for idx in indices_array])
        bm25_scores_array = np.array([bm25_dict.get(idx, bm25_stats['mean']) for idx in indices_array])
        
        # Vectorized normalization
        faiss_norm = np.where(faiss_stats['std'] > 0, 
                            (faiss_scores_array - faiss_stats['mean']) / faiss_stats['std'], 
                            0)
        bm25_norm = np.where(bm25_stats['std'] > 0, 
                           (bm25_scores_array - bm25_stats['mean']) / bm25_stats['std'], 
                           0)
        
        # Vectorized weighted combination
        fused_scores = self.alpha * faiss_norm + (1 - self.alpha) * bm25_norm
        
        # Sort and return top k
        sorted_indices = np.argsort(-fused_scores)[:k]
        return [(int(indices_array[idx]), float(fused_scores[idx])) for idx in sorted_indices]

    def _distribution_fusion_small(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int,
    ) -> List[Tuple[int, float]]:
        """Low-overhead distribution fusion for the common small-result path."""
        faiss_dict = dict(faiss_results)
        bm25_dict = dict(bm25_results)

        faiss_stats = self._calculate_distribution_stats(np.array(list(faiss_dict.values())))
        bm25_stats = self._calculate_distribution_stats(np.array(list(bm25_dict.values())))

        def normalize(score: float, stats: Dict[str, float]) -> float:
            std = stats['std']
            if std <= 0:
                return 0.0
            return (score - stats['mean']) / std

        fused_scores = {
            idx: self.alpha * normalize(faiss_dict.get(idx, faiss_stats['mean']), faiss_stats)
            + (1 - self.alpha) * normalize(bm25_dict.get(idx, bm25_stats['mean']), bm25_stats)
            for idx in faiss_dict.keys() | bm25_dict.keys()
        }

        return heapq.nlargest(k, fused_scores.items(), key=lambda item: item[1])

    def _calculate_distribution_stats(self, scores: np.ndarray) -> Dict[str, float]:
        """Calculate distribution statistics using numpy for efficiency."""
        if len(scores) == 0:
            return {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0}

        # Use numpy's optimized functions
        return {
            'mean': float(scores.mean()),
            'std': float(scores.std()) if len(scores) > 1 else 1.0,
            'min': float(scores.min()),
            'max': float(scores.max())
        }


# Backward compatibility
class HybridSearch(AdvancedHybridSearch):
    """Legacy HybridSearch class for backward compatibility."""

    def __init__(self, faiss_index: FAISSIndex, bm25_index: BM25Index, alpha: float = 0.5):
        super().__init__(faiss_index, bm25_index, FusionStrategy.LINEAR, alpha)
