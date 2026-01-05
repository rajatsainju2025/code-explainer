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

    def build_index(self, codes: List[str]) -> None:
        """Build BM25 index from code snippets."""
        if not HAS_BM25 or BM25Okapi is None:
            raise ImportError("rank_bm25 is not installed; install rank-bm25 to use BM25 retrieval")

        tokenized_corpus = tokenizer.tokenize_list(codes)
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_size = len(codes)

    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar codes using BM25."""
        if self.bm25 is None:
            raise ValueError("BM25 index is not built.")

        tokenized_query = list(tokenizer.tokenize(query))
        scores = self.bm25.get_scores(tokenized_query)
        num_scores = len(scores)

        # Use argpartition for O(n) selection instead of O(n log n) full sort
        k_actual = min(k, num_scores)
        if k_actual == num_scores:
            top_indices = _np_argsort(scores)[::-1]
        else:
            # argpartition is faster for selecting top-k elements
            partition_indices = _np_argpartition(scores, -k_actual)[-k_actual:]
            # Sort only the top-k elements
            top_indices = partition_indices[_np_argsort(scores[partition_indices])[::-1]]
        top_scores = scores[top_indices]

        return top_scores, top_indices

    def get_size(self) -> int:
        """Get the number of documents in the index."""
        return self.corpus_size