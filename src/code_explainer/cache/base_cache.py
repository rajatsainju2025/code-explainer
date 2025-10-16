"""Base caching functionality and interfaces."""

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from .models import CacheConfig, CacheStats
from .utils import ensure_directory


class BaseCache(ABC):
    """Abstract base class for cache implementations."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        ensure_directory(self.cache_dir)
        self._lock = threading.RLock()

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
    """Simple in-memory LRU cache."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []

    def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in memory cache."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Remove least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]

        self._cache[key] = value
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear memory cache."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)