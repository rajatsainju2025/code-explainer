"""Cache factory and consolidation layer to eliminate duplication."""

from typing import Optional, Type, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CacheFactory:
    """Factory for creating cache instances with consistent interface."""

    _cache_registry: Dict[str, Type] = {}

    @classmethod
    def register_cache_type(cls, name: str, cache_class: Type) -> None:
        """Register a cache implementation."""
        cls._cache_registry[name] = cache_class
        logger.debug(f"Registered cache type: {name}")

    @classmethod
    def create_cache(cls, cache_type: str, **kwargs) -> Any:
        """Create a cache instance of specified type."""
        if cache_type not in cls._cache_registry:
            raise ValueError(f"Unknown cache type: {cache_type}. Available: {list(cls._cache_registry.keys())}")
        
        cache_class = cls._cache_registry[cache_type]
        logger.debug(f"Creating cache of type: {cache_type}")
        return cache_class(**kwargs)

    @classmethod
    def get_available_cache_types(cls) -> list:
        """Get list of available cache types."""
        return list(cls._cache_registry.keys())


class CacheBase:
    """Base class for all cache implementations to ensure consistent interface."""

    def __init__(self, cache_dir: str = ".cache", max_size: int = 1000, ttl_seconds: Optional[int] = None):
        """Initialize cache with standard parameters."""
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.size = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError

    def get_size(self) -> int:
        """Get current cache size."""
        return self.size

    def is_full(self) -> bool:
        """Check if cache is at capacity."""
        return self.size >= self.max_size

    def evict_oldest(self) -> bool:
        """Evict oldest entry. Returns True if evicted, False if no entries."""
        raise NotImplementedError


class CacheMetrics:
    """Track cache performance metrics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self) -> None:
        self.misses += 1

    def record_eviction(self) -> None:
        self.evictions += 1

    def record_error(self) -> None:
        self.errors += 1

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Get all metrics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.get_hit_rate(), 4),
            "evictions": self.evictions,
            "errors": self.errors
        }

    def __repr__(self) -> str:
        return f"CacheMetrics(hits={self.hits}, misses={self.misses}, hit_rate={self.get_hit_rate():.2%})"


def create_cache_key(*parts: str, separator: str = ":") -> str:
    """Create consistent cache keys across implementations."""
    return separator.join(str(p) for p in parts if p)
