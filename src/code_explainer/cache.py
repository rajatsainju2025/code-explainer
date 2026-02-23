"""
Caching utilities for code explanations and embeddings.

NOTE: This module has been refactored into a modular structure.
All functionality is now available through the cache package.
"""

# Import everything from the new modular structure for backward compatibility
from .cache import (
    BaseCache,
    CacheConfig,
    CacheStats,
    EmbeddingCache,
    ExplanationCache,
    MemoryCache,
)

__all__ = [
    "ExplanationCache",
    "EmbeddingCache",
    "BaseCache",
    "MemoryCache",
    "CacheConfig",
    "CacheStats",
]
