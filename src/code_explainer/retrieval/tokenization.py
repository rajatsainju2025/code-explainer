"""Text tokenization utilities for retrieval."""

import re
from functools import lru_cache
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor


class TextTokenizer:
    """Tokenizer for text processing in retrieval systems."""
    
    # Use __slots__ to reduce memory footprint
    __slots__ = ('token_pattern', '_executor')

    def __init__(self):
        # Pre-compile regex pattern for better performance
        self.token_pattern = re.compile(r"[^A-Za-z0-9_]+")
        self._executor = None
        self._cache_hits = 0
        self._cache_misses = 0

    @lru_cache(maxsize=4096)  # Doubled from 2048 for better hit rate
    def tokenize(self, text: str) -> Tuple[str, ...]:
        """Tokenize text for BM25 indexing with caching.

        Returns tuple for hashability (required for lru_cache).
        """
        # Fast path for empty/whitespace-only strings
        if not text or text.isspace():
            return ()
        
        # Use filter with None to remove empty strings (faster than list comprehension)
        result = tuple(filter(None, self.token_pattern.split(text.lower())))
        return result

    def tokenize_list(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a list of texts with parallel processing for large batches."""
        n = len(texts)
        
        # For small batches, sequential is faster (no thread overhead)
        # Reduced threshold from 100 to 50 for better parallelization
        if n < 50:
            return [list(self.tokenize(text)) for text in texts]
        
        # For large batches, use parallel processing
        if self._executor is None:
            # Use optimal worker count based on CPU count
            import os
            workers = min(8, os.cpu_count() or 4)
            self._executor = ThreadPoolExecutor(max_workers=workers)
        
        # Process in parallel and convert tuples to lists
        results = list(self._executor.map(self.tokenize, texts))
        return [list(r) for r in results]
    
    def clear_cache(self) -> None:
        """Clear the tokenization cache."""
        self.tokenize.cache_clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        info = self.tokenize.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
        }


# Global tokenizer instance
tokenizer = TextTokenizer()