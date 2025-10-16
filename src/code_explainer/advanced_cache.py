"""
Compatibility module for advanced caching (CacheManager) after modular refactoring.
Provides a unified interface for explanation and advanced (in-memory) caching.
"""
from .cache import ExplanationCache, CacheConfig
from .cache import MemoryCache


class AdvancedCache(MemoryCache):
    """Simple advanced cache with optional tags support."""
    def put(self, key: str, value, tags=None):
        # tags are ignored in this simple implementation
        super().put(key, value)


class CacheManager:
    """Manager for explanation and advanced in-memory caches."""
    def __init__(self, base_cache_dir: str):
        # Initialize explanation cache with default config
        config = CacheConfig(cache_dir=base_cache_dir)
        self._exp_cache = ExplanationCache(base_cache_dir)
        # Advanced cache using in-memory LRU cache
        self._adv_cache = AdvancedCache()

    def get_explanation_cache(self) -> ExplanationCache:
        """Return the explanation cache instance."""
        return self._exp_cache

    def get_advanced_cache(self) -> AdvancedCache:
        """Return the advanced (in-memory) cache instance."""
        return self._adv_cache

    def get_cache_stats(self) -> dict:
        """Return statistics for explanation and advanced caches."""
        # Explanation cache stats as dict
        exp_stats = vars(self._exp_cache.stats())
        # Advanced cache stats: count of entries
        adv_stats = {"memory_entries": self._adv_cache.size()}
        return {"explanation_cache": exp_stats, "advanced_cache": adv_stats}

    def clear_all_caches(self):
        """Clear both explanation and advanced caches."""
        self._exp_cache.clear()
        self._adv_cache.clear()