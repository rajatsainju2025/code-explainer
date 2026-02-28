"""
Caching utilities for code explanations and embeddings.

DEPRECATED: This module is a backward-compatibility shim.
Import directly from ``code_explainer.cache`` (the package) instead.
"""

# Re-export everything from the cache package for backward compatibility
from .cache import (  # noqa: F811
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

import warnings as _warnings
_warnings.warn(
    "Importing from 'code_explainer.cache' (the shim module) is deprecated. "
    "Import from the 'code_explainer.cache' package directly.",
    DeprecationWarning,
    stacklevel=2,
)
