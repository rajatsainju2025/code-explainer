"""BM25 index management for lexical search."""

from typing import Any, List, Optional, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None

from .tokenization import tokenizer

# Pre-cache numpy functions for micro-optimization
_np_argsort = np.argsort
_np_argpartition = np.argpartition


class BM25Index:
    """Manages BM25 index for lexical similarity search."""

    __slots__ = ("bm25", "corpus_size")

    def __init__(self):
        if not HAS_BM25:
            raise ImportError("rank_bm25 is not installed; install rank-bm25 to use BM25 retrieval")

        self.bm25: Optional[Any] = None
        self.corpus_size = 0

    def build_index(self, codes: List[str], batch_size: int = 1000) -> None:
        """Build BM25 index from code snippets with optimized batching.
        
        Args:
            codes: List of code snippets to index
            batch_size: Size of batches for tokenization (for large corpora)
        """
        if not HAS_BM25 or BM25Okapi is None:
            raise ImportError("rank_bm25 is not installed; install rank-bm25 to use BM25 retrieval")

        # For large corpora, tokenize in batches to reduce memory peaks
        if len(codes) > batch_size:
            tokenized_corpus = []
            for i in range(0, len(codes), batch_size):
                batch = codes[i:i + batch_size]
                tokenized_corpus.extend(tokenizer.tokenize_list(batch))
        else:
            tokenized_corpus = tokenizer.tokenize_list(codes)
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_size = len(codes)

    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar codes using BM25."""
        if self.bm25 is None:
            raise ValueError("BM25 index is not built.")

        tokenized_query = list(tokenizer.tokenize(query))
        
        # Fast path for empty query
        if not tokenized_query:
            return np.array([]), np.array([])
        
        scores = self.bm25.get_scores(tokenized_query)
        num_scores = len(scores)
        
        # Fast path for requesting all or more than available
        k_actual = min(k, num_scores)
        if k_actual <= 0:
            return np.array([]), np.array([])
        
        if k_actual >= num_scores:
            # Return all scores sorted
            top_indices = _np_argsort(scores)[::-1]
            return scores[top_indices], top_indices
        
        # Use argpartition for O(n) selection instead of O(n log n) full sort
        # This is significantly faster for large corpora when k << n
        partition_indices = _np_argpartition(scores, -k_actual)[-k_actual:]
        # Sort only the top-k elements using in-place operations
        top_k_scores = scores[partition_indices]  # View, not copy
        sorted_order = _np_argsort(top_k_scores)[::-1]
        top_indices = partition_indices[sorted_order]
        top_scores = top_k_scores[sorted_order]

        return top_scores, top_indices

    def get_size(self) -> int:
        """Get the number of documents in the index."""
        return self.corpus_size