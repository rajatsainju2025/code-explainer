"""Memory optimization utilities and utilities for efficient code.

Optimized for:
- Cross-platform memory monitoring (macOS, Linux, Windows)
- Efficient LRU eviction without full cache clear
- Lazy module loading with caching
- Configurable garbage collection
"""

from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
import sys
import gc
import platform

# Detect platform once at module load (cache results)
_PLATFORM = platform.system()
_IS_LINUX = _PLATFORM == "Linux"
_IS_MACOS = _PLATFORM == "Darwin"
_IS_WINDOWS = _PLATFORM == "Windows"

# Cache psutil module reference if available (avoid repeated imports)
_psutil = None

def _get_psutil():\n    \"\"\"Lazily import and cache psutil module.\"\"\"\n    global _psutil\n    if _psutil is None:\n        try:\n            import psutil\n            _psutil = psutil\n        except ImportError:\n            _psutil = False  # Mark as unavailable\n    return _psutil if _psutil is not False else None


class LazyLoader:
    """Lazy load modules to reduce startup memory.
    
    Uses __slots__ for memory efficiency.
    """
    __slots__ = ('module_name', '_module')
    
    def __init__(self, module_name: str):
        """Initialize lazy loader.
        
        Args:
            module_name: Module to lazily load
        """
        object.__setattr__(self, 'module_name', module_name)
        object.__setattr__(self, '_module', None)
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from lazily loaded module.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute from module
        """
        module = object.__getattribute__(self, '_module')
        if module is None:
            module_name = object.__getattribute__(self, 'module_name')
            __import__(module_name)
            module = sys.modules[module_name]
            object.__setattr__(self, '_module', module)
        return getattr(module, name)


class MemoryOptimizedCache:
    """LRU Cache with memory optimization using OrderedDict.
    
    Uses __slots__ and OrderedDict for efficient LRU eviction
    without clearing the entire cache.
    """
    
    __slots__ = ('_cache', '_max_size', '_hits', '_misses')
    
    def __init__(self, max_size: int = 1000):
        """Initialize optimized cache.
        
        Args:
            max_size: Maximum cache size
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def set(self, key: str, value: Any) -> None:
        """Set cache entry with LRU eviction.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # If key exists, move to end and update
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        
        # Evict oldest entries if at capacity (LRU eviction)
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cache entry and update access order.
        
        Args:
            key: Cache key
            default: Default if not found
            
        Returns:
            Cached value or default
        """
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)  # Update LRU order
            return self._cache[key]
        self._misses += 1
        return default
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists without updating LRU order."""
        return key in self._cache
    
    def memory_info(self) -> Dict[str, Any]:
        """Get memory information and statistics.
        
        Returns:
            Dictionary with memory metrics
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "cache_size": len(self._cache),
            "max_size": self._max_size,
            "approx_bytes": sys.getsizeof(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4)
        }


def get_memory_usage() -> int:
    """Get current memory usage in bytes (cross-platform).
    
    Returns:
        Current process memory usage in bytes
    """
    try:
        if _IS_LINUX:
            # Linux: read from /proc (fastest)
            import os
            with open(f'/proc/{os.getpid()}/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) * 1024
        elif _IS_MACOS or _IS_WINDOWS:
            # macOS/Windows: use psutil if available
            try:
                import psutil
                return psutil.Process().memory_info().rss
            except ImportError:
                pass
            
            # macOS fallback: use resource module
            if _IS_MACOS:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (FileNotFoundError, ValueError, OSError, ImportError):
        pass
    
    return 0


def get_memory_info() -> Dict[str, Any]:
    """Get detailed memory information.
    
    Returns:
        Dictionary with detailed memory metrics
    """
    memory_bytes = get_memory_usage()
    
    return {
        "memory_bytes": memory_bytes,
        "memory_mb": round(memory_bytes / (1024 * 1024), 2),
        "platform": _PLATFORM,
        "gc_enabled": gc.isenabled(),
        "gc_counts": gc.get_count()
    }


def optimize_memory(aggressive: bool = False) -> Tuple[int, int]:
    """Trigger garbage collection and memory optimization.
    
    Args:
        aggressive: If True, run multiple GC passes
        
    Returns:
        Tuple of (objects_collected, memory_freed_bytes)
    """
    memory_before = get_memory_usage()
    
    if aggressive:
        # Multiple passes for thorough cleanup
        collected = 0
        for _ in range(3):
            collected += gc.collect()
    else:
        collected = gc.collect()
    
    memory_after = get_memory_usage()
    memory_freed = max(0, memory_before - memory_after)
    
    return collected, memory_freed


def optimize_memory() -> None:
    """Trigger garbage collection and memory optimization."""
    gc.collect()
