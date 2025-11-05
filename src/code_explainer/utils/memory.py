"""Memory optimization utilities and utilities for efficient code."""

from typing import Any, Optional, Type, Dict
import sys
import gc


class LazyLoader:
    """Lazy load modules to reduce startup memory."""
    
    def __init__(self, module_name: str):
        """Initialize lazy loader.
        
        Args:
            module_name: Module to lazily load
        """
        self.module_name = module_name
        self._module: Optional[Any] = None
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from lazily loaded module.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute from module
        """
        if self._module is None:
            __import__(self.module_name)
            self._module = sys.modules[self.module_name]
        return getattr(self._module, name)


class MemoryOptimizedCache:
    """Cache with memory optimization."""
    
    __slots__ = ('_cache', '_max_size')
    
    def __init__(self, max_size: int = 1000):
        """Initialize optimized cache.
        
        Args:
            max_size: Maximum cache size
        """
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
    
    def set(self, key: str, value: Any) -> None:
        """Set cache entry.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if len(self._cache) >= self._max_size:
            self._cache.clear()
            gc.collect()
        self._cache[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cache entry.
        
        Args:
            key: Cache key
            default: Default if not found
            
        Returns:
            Cached value or default
        """
        return self._cache.get(key, default)
    
    def memory_info(self) -> Dict[str, int]:
        """Get memory information.
        
        Returns:
            Dictionary with memory metrics
        """
        return {
            "cache_size": len(self._cache),
            "max_size": self._max_size,
            "approx_bytes": sys.getsizeof(self._cache)
        }


def get_memory_usage() -> int:
    """Get current memory usage in bytes.
    
    Returns:
        Current process memory usage
    """
    import os
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return 0


def optimize_memory() -> None:
    """Trigger garbage collection and memory optimization."""
    gc.collect()
