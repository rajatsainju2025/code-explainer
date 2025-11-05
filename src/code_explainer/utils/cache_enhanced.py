"""Enhanced caching infrastructure with statistics and warmup."""

from typing import Callable, Any, Dict, Optional
import hashlib
import time


class CacheStatistics:
    """Track cache performance statistics."""
    
    __slots__ = ('hits', 'misses', 'evictions', 'access_times')
    
    def __init__(self):
        """Initialize cache statistics."""
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.access_times: list = []
    
    def record_hit(self, access_time: float) -> None:
        """Record cache hit.
        
        Args:
            access_time: Time taken to access
        """
        self.hits += 1
        self.access_times.append(access_time)
    
    def record_miss(self) -> None:
        """Record cache miss."""
        self.misses += 1
    
    def hit_rate(self) -> float:
        """Get cache hit rate.
        
        Returns:
            Hit rate as percentage
        """
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0
    
    def avg_access_time(self) -> float:
        """Get average access time.
        
        Returns:
            Average access time in seconds
        """
        return sum(self.access_times) / len(self.access_times) if self.access_times else 0


class EnhancedCache:
    """Cache with statistics and warmup capability."""
    
    def __init__(self, max_entries: int = 1000):
        """Initialize enhanced cache.
        
        Args:
            max_entries: Maximum cache entries
        """
        self._cache: Dict[str, Any] = {}
        self._stats = CacheStatistics()
        self._max_entries = max_entries
        self._access_order: list = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        start = time.time()
        if key in self._cache:
            access_time = time.time() - start
            self._stats.record_hit(access_time)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        self._stats.record_miss()
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cache entry.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_entries:
            # LRU eviction
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
            self._stats.evictions += 1
        
        self._cache[key] = value
        self._access_order.append(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate(),
            "evictions": self._stats.evictions,
            "avg_access_time": self._stats.avg_access_time(),
            "current_size": len(self._cache),
            "max_size": self._max_entries
        }
    
    def warmup(self, entries: Dict[str, Any]) -> None:
        """Warm up cache with entries.
        
        Args:
            entries: Dictionary of entries to preload
        """
        for key, value in entries.items():
            self.set(key, value)
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()
        self._stats = CacheStatistics()


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Hash-based cache key
    """
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()
