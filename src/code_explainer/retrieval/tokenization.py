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

    @lru_cache(maxsize=2048)
    def tokenize(self, text: str) -> Tuple[str, ...]:
        """Tokenize text for BM25 indexing with caching.

        Returns tuple for hashability (required for lru_cache).
        """
        # Use filter with None to remove empty strings (faster than list comprehension)
        return tuple(filter(None, self.token_pattern.split(text)))

    def tokenize_list(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a list of texts with parallel processing for large batches."""
        n = len(texts)
        
        # For small batches, sequential is faster (no thread overhead)
        if n < 100:
            return [list(self.tokenize(text)) for text in texts]
        
        # For large batches, use parallel processing
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Process in parallel and convert tuples to lists
        results = list(self._executor.map(self.tokenize, texts))
        return [list(r) for r in results]
    
    def clear_cache(self) -> None:
        """Clear the tokenization cache."""
        self.tokenize.cache_clear()


# Global tokenizer instance
tokenizer = TextTokenizer()