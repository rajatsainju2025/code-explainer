"""Cache module for code explanations and embeddings."""

from .base_cache import (BaseCache, MemoryCache, AsyncCacheMixin,
                        EvictionPolicy, CacheEntry)
from .embedding_cache import EmbeddingCache
from .explanation_cache import ExplanationCache
from .models import CacheConfig, CacheEntry as BaseCacheEntry, CacheStats, EmbeddingEntry, ExplanationEntry
from .utils import (calculate_cache_score, compress_data, decompress_data,
                   ensure_directory, generate_cache_key, is_expired, safe_file_operation)

__all__ = [
    "AsyncCacheMixin",
    "BaseCache",
    "CacheConfig",
    "CacheEntry",
    "BaseCacheEntry",
    "CacheStats",
    "EmbeddingCache",
    "EmbeddingEntry",
    "EvictionPolicy",
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