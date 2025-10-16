"""Text tokenization utilities for retrieval."""

import re
from functools import lru_cache
from typing import Tuple


class TextTokenizer:
    """Tokenizer for text processing in retrieval systems."""

    def __init__(self):
        # Pre-compile regex pattern for better performance
        self.token_pattern = re.compile(r"[^A-Za-z0-9_]+")

    @lru_cache(maxsize=1024)
    def tokenize(self, text: str) -> Tuple[str, ...]:
        """Tokenize text for BM25 indexing with caching.

        Returns tuple for hashability (required for lru_cache).
        """
        tokens = [t for t in self.token_pattern.split(text) if t]
        return tuple(tokens)

    def tokenize_list(self, texts: list) -> list:
        """Tokenize a list of texts."""
        return [list(self.tokenize(text)) for text in texts]


# Global tokenizer instance
tokenizer = TextTokenizer()