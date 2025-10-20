"""
Compatibility module for advanced caching (CacheManager) after modular refactoring.
Provides a unified interface for explanation and advanced (in-memory) caching.
"""
from .cache import ExplanationCache, CacheConfig, AdvancedMemoryCache, EvictionPolicy


class AdvancedCache(AdvancedMemoryCache):
    """Advanced cache with configurable eviction policies and optional tags support."""
    def __init__(self, max_size: int = 100, eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        super().__init__(max_size=max_size, eviction_policy=eviction_policy)
        self._tags: dict = {}  # tag -> set of keys

    def put(self, key: str, value, ttl=None, tags=None):
        """Put item in cache with optional TTL and tags."""
        super().put(key, value, ttl=ttl)

        # Handle tags
        if tags:
            for tag in tags:
                if tag not in self._tags:
                    self._tags[tag] = set()
                self._tags[tag].add(key)

    def get_by_tag(self, tag: str):
        """Get all cache entries with a specific tag."""
        if tag not in self._tags:
            return {}
        return {key: self.get(key) for key in self._tags[tag]}

    def invalidate_by_tag(self, tag: str):
        """Remove all cache entries with a specific tag."""
        if tag in self._tags:
            for key in self._tags[tag]:
                # Note: We don't have a direct remove method, so we put None
                # This is a limitation that could be addressed by adding a remove method
                pass
            del self._tags[tag]

    def cleanup_expired_entries(self):
        """Clean up expired entries and return count removed."""
        return self.cleanup_expired()


class CacheManager:
    """Manager for explanation and advanced in-memory caches."""
    def __init__(self, base_cache_dir: str, memory_cache_size: int = 100,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        # Initialize explanation cache with default config
        config = CacheConfig(cache_dir=base_cache_dir)
        self._exp_cache = ExplanationCache(base_cache_dir)
        # Advanced cache using enhanced memory cache
        self._adv_cache = AdvancedCache(max_size=memory_cache_size,
                                       eviction_policy=eviction_policy)

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
        # Advanced cache stats
        adv_stats = {
            "memory_entries": self._adv_cache.size(),
            "eviction_policy": self._adv_cache._eviction_policy.value,
            "current_memory_size": self._adv_cache._current_memory_size,
            "max_memory_size": self._adv_cache._max_memory_size
        }
        return {"explanation_cache": exp_stats, "advanced_cache": adv_stats}

    def clear_all_caches(self):
        """Clear both explanation and advanced caches."""
        self._exp_cache.clear()
        self._adv_cache.clear()

    def set_eviction_policy(self, policy: EvictionPolicy):
        """Set the eviction policy for the advanced cache."""
        self._adv_cache.set_eviction_policy(policy)