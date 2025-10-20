"""Hybrid search combining multiple retrieval methods with advanced features."""

import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
import re

import numpy as np

from .bm25_index import BM25Index
from .faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Strategies for fusing multiple retrieval scores."""
    LINEAR = "linear"  # Simple weighted linear combination
    RRF = "rrf"  # Reciprocal Rank Fusion
    DISTRIBUTION_BASED = "distribution_based"  # Distribution-based fusion


class QueryExpansion:
    """Query expansion utilities."""

    @staticmethod
    def expand_with_synonyms(query: str, synonyms: Optional[Dict[str, List[str]]] = None) -> List[str]:
        """Expand query with synonyms."""
        if not synonyms:
            return [query]

        expanded_queries = [query]
        words = re.findall(r'\b\w+\b', query.lower())

        for word in words:
            if word in synonyms:
                for synonym in synonyms[word][:2]:  # Limit to 2 synonyms per word
                    expanded = query.lower().replace(word, synonym)
                    if expanded not in expanded_queries:
                        expanded_queries.append(expanded)

        return expanded_queries[:5]  # Limit to 5 expanded queries

    @staticmethod
    def expand_with_ngrams(query: str) -> List[str]:
        """Generate n-gram expansions."""
        words = query.split()
        expansions = [query]

        # Add bigrams
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram not in query:
                    expansions.append(bigram)

        return expansions


class AdvancedHybridSearch:
    """Advanced hybrid search with multiple fusion strategies and query expansion."""

    def __init__(self,
                 faiss_index: Optional[FAISSIndex] = None,
                 bm25_index: Optional[BM25Index] = None,
                 fusion_strategy: FusionStrategy = FusionStrategy.LINEAR,
                 alpha: float = 0.5,
                 rrf_k: int = 60):
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.query_expander = QueryExpansion()

    def search(self,
               query: str,
               k: int,
               expand_query: bool = False,
               synonyms: Optional[Dict[str, List[str]]] = None) -> List[Tuple[int, float]]:
        """Perform advanced hybrid search with optional query expansion."""
        if expand_query:
            queries = self.query_expander.expand_with_synonyms(query, synonyms)
            logger.info(f"Expanded query '{query}' to {len(queries)} variations")
        else:
            queries = [query]

        all_results = []
        for q in queries:
            results = self._search_single_query(q, k * 2)  # Get more candidates for fusion
            all_results.append(results)

        # Combine results from all query variations
        return self._combine_expanded_results(all_results, k)

    def _search_single_query(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Search with a single query using configured fusion strategy."""
        faiss_results = self._faiss_search(query, k)
        bm25_results = self._bm25_search(query, k)

        if not faiss_results and not bm25_results:
            return []

        if self.fusion_strategy == FusionStrategy.LINEAR:
            return self._linear_fusion(faiss_results, bm25_results, k)
        elif self.fusion_strategy == FusionStrategy.RRF:
            return self._rrf_fusion(faiss_results, bm25_results, k)
        elif self.fusion_strategy == FusionStrategy.DISTRIBUTION_BASED:
            return self._distribution_fusion(faiss_results, bm25_results, k)
        else:
            return self._linear_fusion(faiss_results, bm25_results, k)

    def _faiss_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform FAISS search."""
        if not self.faiss_index:
            return []

        try:
            distances, indices = self.faiss_index.search(query, k)
            results = []
            for d, i in zip(distances[0], indices[0]):
                # Convert distance to similarity score (higher is better)
                similarity = 1.0 / (1.0 + float(d))
                results.append((int(i), similarity))
            return results
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}")
            return []

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform BM25 search."""
        if not self.bm25_index:
            return []

        try:
            scores, indices = self.bm25_index.search(query, k)
            return [(int(i), float(s)) for s, i in zip(scores, indices)]
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    def _linear_fusion(self,
                      faiss_results: List[Tuple[int, float]],
                      bm25_results: List[Tuple[int, float]],
                      k: int) -> List[Tuple[int, float]]:
        """Linear fusion of FAISS and BM25 results."""
        # Convert to dictionaries for easy merging
        faiss_dict = dict(faiss_results)
        bm25_dict = dict(bm25_results)

        # Normalize BM25 scores to [0, 1] range
        if bm25_dict:
            bm_scores = list(bm25_dict.values())
            bm_min, bm_max = min(bm_scores), max(bm_scores)
            bm_range = bm_max - bm_min if bm_max > bm_min else 1.0
            bm25_dict = {i: (s - bm_min) / bm_range for i, s in bm25_dict.items()}

        # Fuse scores
        all_indices = set(faiss_dict.keys()) | set(bm25_dict.keys())
        fused_results = []

        for idx in all_indices:
            faiss_score = faiss_dict.get(idx, 0.0)
            bm25_score = bm25_dict.get(idx, 0.0)
            fused_score = self.alpha * faiss_score + (1 - self.alpha) * bm25_score
            fused_results.append((idx, fused_score))

        # Sort by fused score and return top k
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:k]

    def _rrf_fusion(self,
                   faiss_results: List[Tuple[int, float]],
                   bm25_results: List[Tuple[int, float]],
                   k: int) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion."""
        rrf_scores: Dict[int, float] = {}

        # Add FAISS results with RRF
        for rank, (idx, _) in enumerate(faiss_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))

        # Add BM25 results with RRF
        for rank, (idx, _) in enumerate(bm25_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort by RRF score
        fused_results = [(idx, score) for idx, score in rrf_scores.items()]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:k]

    def _distribution_fusion(self,
                           faiss_results: List[Tuple[int, float]],
                           bm25_results: List[Tuple[int, float]],
                           k: int) -> List[Tuple[int, float]]:
        """Distribution-based fusion using score distributions."""
        faiss_scores = [score for _, score in faiss_results] if faiss_results else []
        bm25_scores = [score for _, score in bm25_results] if bm25_results else []

        # Calculate distribution statistics
        faiss_stats = self._calculate_distribution_stats(faiss_scores)
        bm25_stats = self._calculate_distribution_stats(bm25_scores)

        # Fuse using distribution-aware weighting
        faiss_dict = dict(faiss_results)
        bm25_dict = dict(bm25_results)
        all_indices = set(faiss_dict.keys()) | set(bm25_dict.keys())

        fused_results = []
        for idx in all_indices:
            faiss_score = faiss_dict.get(idx, faiss_stats['mean'])
            bm25_score = bm25_dict.get(idx, bm25_stats['mean'])

            # Normalize using distribution stats
            faiss_norm = (faiss_score - faiss_stats['mean']) / faiss_stats['std'] if faiss_stats['std'] > 0 else 0
            bm25_norm = (bm25_score - bm25_stats['mean']) / bm25_stats['std'] if bm25_stats['std'] > 0 else 0

            # Weighted combination
            fused_score = self.alpha * faiss_norm + (1 - self.alpha) * bm25_norm
            fused_results.append((idx, fused_score))

        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:k]

    def _calculate_distribution_stats(self, scores: List[float]) -> Dict[str, float]:
        """Calculate distribution statistics for a list of scores."""
        if not scores:
            return {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0}

        scores_array = np.array(scores)
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)) or 1.0,
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array))
        }

    def _combine_expanded_results(self,
                                all_results: List[List[Tuple[int, float]]],
                                k: int) -> List[Tuple[int, float]]:
        """Combine results from multiple query expansions."""
        if len(all_results) == 1:
            return all_results[0][:k]

        # Use RRF to combine results from different query expansions
        rrf_scores: Dict[int, float] = {}

        for results in all_results:
            for rank, (idx, _) in enumerate(results, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))

        combined_results = [(idx, score) for idx, score in rrf_scores.items()]
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]

    def set_fusion_strategy(self, strategy: FusionStrategy):
        """Change the fusion strategy."""
        self.fusion_strategy = strategy

    def set_alpha(self, alpha: float):
        """Change the fusion weight alpha."""
        self.alpha = max(0.0, min(1.0, alpha))


# Backward compatibility
class HybridSearch(AdvancedHybridSearch):
    """Legacy HybridSearch class for backward compatibility."""

    def __init__(self, faiss_index: FAISSIndex, bm25_index: BM25Index, alpha: float = 0.5):
        super().__init__(faiss_index, bm25_index, FusionStrategy.LINEAR, alpha)
