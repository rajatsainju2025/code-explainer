"""Cache module for code explanations and embeddings."""

from .base_cache import BaseCache, MemoryCache
from .embedding_cache import EmbeddingCache
from .explanation_cache import ExplanationCache
from .models import CacheConfig, CacheEntry, CacheStats, EmbeddingEntry, ExplanationEntry
from .utils import (calculate_cache_score, compress_data, decompress_data,
                   ensure_directory, generate_cache_key, is_expired, safe_file_operation)

__all__ = [
    "BaseCache",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "EmbeddingCache",
    "EmbeddingEntry",
    "ExplanationCache",
    "ExplanationEntry",
    "MemoryCache",
    "calculate_cache_score",
    "compress_data",
    "decompress_data",
    "ensure_directory",
    "generate_cache_key",
    "is_expired",
    "safe_file_operation",
]