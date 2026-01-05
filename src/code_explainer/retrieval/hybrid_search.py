"""Hybrid search combining multiple retrieval methods with advanced features."""

import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
import re

import numpy as np

from .bm25_index import BM25Index
from .faiss_index import FAISSIndex

logger = logging.getLogger(__name__)

# Pre-compile regex for query tokenization
_WORD_PATTERN = re.compile(r'\b\w+\b')


class FusionStrategy(Enum):
    """Strategies for fusing multiple retrieval scores."""
    LINEAR = "linear"  # Simple weighted linear combination
    RRF = "rrf"  # Reciprocal Rank Fusion
    DISTRIBUTION_BASED = "distribution_based"  # Distribution-based fusion


class QueryExpansion:
    """Query expansion utilities."""
    
    __slots__ = ()  # No instance attributes needed

    @staticmethod
    def expand_with_synonyms(query: str, synonyms: Optional[Dict[str, List[str]]] = None) -> List[str]:
        """Expand query with synonyms."""
        if not synonyms:
            return [query]

        expanded_queries = [query]
        query_lower = query.lower()
        words = _WORD_PATTERN.findall(query_lower)

        for word in words:
            word_synonyms = synonyms.get(word)
            if word_synonyms:
                for synonym in word_synonyms[:2]:  # Limit to 2 synonyms per word
                    expanded = query_lower.replace(word, synonym)
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
    
    __slots__ = ('faiss_index', 'bm25_index', 'fusion_strategy', 'alpha', 'rrf_k', 'query_expander')

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
            if len(queries) > 1:
                logger.debug("Expanded query '%s' to %d variations", query, len(queries))
        else:
            queries = [query]

        # Fast path for single query
        if len(queries) == 1:
            return self._search_single_query(queries[0], k)

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

    def _rrf_fusion(self,
                   faiss_results: List[Tuple[int, float]],
                   bm25_results: List[Tuple[int, float]],
                   k: int) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion with vectorized operations."""
        rrf_scores: Dict[int, float] = {}

        # Vectorized RRF calculation for FAISS results
        if faiss_results:
            faiss_indices = np.array([idx for idx, _ in faiss_results])
            faiss_ranks = np.arange(1, len(faiss_results) + 1)
            faiss_rrf_scores = 1.0 / (self.rrf_k + faiss_ranks)
            for idx, score in zip(faiss_indices, faiss_rrf_scores):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + score

        # Vectorized RRF calculation for BM25 results
        if bm25_results:
            bm25_indices = np.array([idx for idx, _ in bm25_results])
            bm25_ranks = np.arange(1, len(bm25_results) + 1)
            bm25_rrf_scores = 1.0 / (self.rrf_k + bm25_ranks)
            for idx, score in zip(bm25_indices, bm25_rrf_scores):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + score

        # Sort by RRF score
        fused_results = [(idx, score) for idx, score in rrf_scores.items()]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:k]

    def _distribution_fusion(self,
                           faiss_results: List[Tuple[int, float]],
                           bm25_results: List[Tuple[int, float]],
                           k: int) -> List[Tuple[int, float]]:
        """Distribution-based fusion using score distributions with numpy optimization."""
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
