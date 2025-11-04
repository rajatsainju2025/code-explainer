"""Base caching functionality and interfaces."""

import asyncio
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
    """Enhanced cache entry with eviction metadata."""
    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.ttl = ttl
        self.size = self._calculate_size()

    def _calculate_size(self) -> int:
        """Calculate approximate memory size of the entry."""
        # Rough estimation - can be enhanced for more accuracy
        return len(str(self.value)) + len(self.key) + 100  # overhead

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


class AdvancedMemoryCache(AsyncCacheMixin):
    """Enhanced memory cache with advanced eviction policies and async support."""

    def __init__(self, max_size: int = 100, eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self._max_size = max_size
        self._eviction_policy = eviction_policy
        self._entries: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict[str, None] = OrderedDict()
        self._frequency: Dict[str, int] = {}
        self._ttl_heap: list = []  # For TTL-based eviction
        self._size_heap: list = []  # For size-based eviction
        self._current_memory_size = 0
        self._max_memory_size = max_size * 1024 * 1024  # Assume average entry size
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache with advanced eviction tracking."""
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                if entry.is_expired():
                    self._remove_entry(key)
                    return None

                entry.access()

                # Update access order based on policy
                if self._eviction_policy == EvictionPolicy.LRU:
                    self._access_order.move_to_end(key)
                elif self._eviction_policy == EvictionPolicy.LFU:
                    self._frequency[key] = self._frequency.get(key, 0) + 1

                return entry.value
        return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in memory cache with advanced eviction."""
        with self._lock:
            entry = CacheEntry(key, value, ttl)

            # Remove existing entry if present
            if key in self._entries:
                self._remove_entry(key)

            # Check if we need to evict before adding
            while self._should_evict():
                self._evict_one_entry()

            # Add new entry
            self._entries[key] = entry
            self._current_memory_size += entry.size

            # Update data structures based on policy
            if self._eviction_policy == EvictionPolicy.LRU:
                self._access_order[key] = None
            elif self._eviction_policy == EvictionPolicy.LFU:
                self._frequency[key] = 1
            elif self._eviction_policy == EvictionPolicy.TTL and ttl is not None:
                heapq.heappush(self._ttl_heap, (entry.created_at + ttl, key))
            elif self._eviction_policy == EvictionPolicy.SIZE:
                heapq.heappush(self._size_heap, (entry.size, key))

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from all data structures."""
        if key in self._entries:
            entry = self._entries[key]
            self._current_memory_size -= entry.size
            del self._entries[key]

        if key in self._access_order:
            del self._access_order[key]
        if key in self._frequency:
            del self._frequency[key]

    def _evict_one_entry(self) -> None:
        """Evict one entry based on the current eviction policy."""
        if not self._entries:
            return

        key_to_evict = None

        if self._eviction_policy == EvictionPolicy.LRU:
            key_to_evict = next(iter(self._access_order))
        elif self._eviction_policy == EvictionPolicy.LFU:
            # Find key with minimum frequency
            if self._frequency:
                key_to_evict = min(self._frequency.keys(), key=lambda k: self._frequency[k])
        elif self._eviction_policy == EvictionPolicy.FIFO:
            key_to_evict = next(iter(self._access_order))
        elif self._eviction_policy == EvictionPolicy.TTL:
            # Clean expired entries
            current_time = time.time()
            while self._ttl_heap and self._ttl_heap[0][0] <= current_time:
                _, expired_key = heapq.heappop(self._ttl_heap)
                if expired_key in self._entries and self._entries[expired_key].is_expired():
                    key_to_evict = expired_key
                    break
            if key_to_evict is None:
                key_to_evict = next(iter(self._access_order))  # Fallback to LRU
        elif self._eviction_policy == EvictionPolicy.SIZE:
            if self._size_heap:
                _, key_to_evict = heapq.heappop(self._size_heap)

        if key_to_evict:
            self._remove_entry(key_to_evict)

    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        return len(self._entries) >= self._max_size or self._current_memory_size > self._max_memory_size

    def clear(self) -> None:
        """Clear memory cache."""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()
            self._frequency.clear()
            self._ttl_heap.clear()
            self._size_heap.clear()
            self._current_memory_size = 0

    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._entries)

    def cleanup_expired(self) -> int:
        """Clean up expired entries. Returns number of entries removed."""
        with self._lock:
            expired_keys = [key for key, entry in self._entries.items() if entry.is_expired()]
            # Use generator for efficient removal
            removed_count = 0
            for key in (key for key, entry in self._entries.items() if entry.is_expired()):
                self._remove_entry(key)
                removed_count += 1
            return removed_count

    def set_eviction_policy(self, policy: EvictionPolicy) -> None:
        """Change the eviction policy."""
        with self._lock:
            self._eviction_policy = policy
            # Rebuild data structures if needed
            if policy == EvictionPolicy.LRU:
                self._access_order = OrderedDict((k, None) for k in self._entries.keys())
            elif policy == EvictionPolicy.LFU:
                self._frequency = {k: self._entries[k].access_count for k in self._entries.keys()}