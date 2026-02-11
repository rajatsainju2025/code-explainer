"""Base caching functionality and interfaces."""

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from collections import OrderedDict
import heapq

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
