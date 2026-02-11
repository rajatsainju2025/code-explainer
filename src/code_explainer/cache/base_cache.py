"""Base caching functionality and interfaces."""

import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
from collections import OrderedDict
import heapq

from .models import CacheConfig, CacheStats
from .utils import ensure_directory


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based eviction


class CacheEntry:
    """Enhanced cache entry with eviction metadata (optimized with __slots__)."""
    __slots__ = ('key', 'value', 'created_at', 'last_accessed', 'access_count', 'ttl', '_size')
    
    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.ttl = ttl
        self._size: Optional[int] = None  # Lazy calculation

    @property
    def size(self) -> int:
        """Calculate approximate memory size lazily."""
        if self._size is None:
            # Rough estimation - optimized for speed
            value_size = len(str(self.value)) if not isinstance(self.value, (int, float, bool)) else 8
            self._size = value_size + len(self.key) + 80  # reduced overhead estimate
        return self._size

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def access(self):
        """Mark entry as accessed."""
        self.last_accessed = time.time()
        self.access_count += 1

    def __lt__(self, other):
        """Comparison for heap operations."""
        if not isinstance(other, CacheEntry):
            return NotImplemented
        return self.last_accessed < other.last_accessed


class AsyncCacheMixin:
    """Mixin to add async support to cache implementations."""

    async def aget(self, *args, **kwargs) -> Optional[Any]:
        """Async get operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, getattr(self, 'get'), *args, **kwargs)

    async def aput(self, *args, **kwargs) -> None:
        """Async put operation."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, getattr(self, 'put'), *args, **kwargs)

    async def aclear(self) -> None:
        """Async clear operation."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, getattr(self, 'clear'))

    async def asize(self) -> int:
        """Async size operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, getattr(self, 'size'))

    async def astats(self) -> CacheStats:
        """Async stats operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, getattr(self, 'stats'))


class BaseCache(ABC, AsyncCacheMixin):
    """Abstract base class for cache implementations."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        ensure_directory(self.cache_dir)
        self._lock = threading.RLock()
        self._eviction_policy = EvictionPolicy.LRU
        self._max_memory_size = getattr(config, 'memory_cache_size', 100) * 1024 * 1024  # Convert to bytes
        self._current_memory_size = 0

    @abstractmethod
    def get(self, *args, **kwargs) -> Optional[Any]:
        """Get an item from cache."""
        pass

    @abstractmethod
    def put(self, *args, **kwargs) -> None:
        """Put an item in cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached items."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get the number of cached items."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        pass

    def set_eviction_policy(self, policy: EvictionPolicy) -> None:
        """Set the eviction policy for the cache."""
        self._eviction_policy = policy

    def _should_evict(self) -> bool:
        """Check if eviction is needed based on memory size."""
        return self._current_memory_size > self._max_memory_size

    def _evict_entries(self, target_size: Optional[int] = None) -> None:
        """Evict entries based on the current eviction policy."""
        if target_size is None:
            target_size = int(self._max_memory_size * 0.8)  # Target 80% of max size

        # This should be implemented by subclasses that maintain entry lists
        pass

    def _acquire_lock(self):
        """Acquire the cache lock."""
        return self._lock.__enter__()

    def _release_lock(self):
        """Release the cache lock."""
        return self._lock.__exit__(None, None, None)

    def __enter__(self):
        """Context manager entry."""
        self._acquire_lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._release_lock()


class MemoryCache:
    """Simple in-memory LRU cache (backward compatibility)."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in memory cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            # Remove least recently used
            self._cache.popitem(last=False)

        self._cache[key] = value

    def clear(self) -> None:
        """Clear memory cache."""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)
