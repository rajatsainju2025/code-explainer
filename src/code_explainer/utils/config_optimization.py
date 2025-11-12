"""Configuration optimization and memoization utilities.

This module provides efficient configuration management with caching
and memoization to reduce repeated lookups and computations.
"""

from typing import Any, Dict, Optional, Callable, Tuple
import threading
import functools
import json


class ConfigCache:
    """Efficient configuration caching with TTL."""
    
    __slots__ = ('_cache', '_lock', '_ttl')
    
    def __init__(self, ttl: float = 3600):
        """Initialize config cache."""
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached config value."""
        import time
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return value
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Cache config value."""
        import time
        with self._lock:
            self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()


class FastConfig:
    """Configuration object with optimized access."""
    
    __slots__ = ('_config', '_cache', '_lock')
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fast config."""
        self._config = config
        self._cache = ConfigCache()
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with caching."""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        
        # Navigate nested keys (e.g., "db.host")
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = None
                break
        
        if value is None:
            value = default
        else:
            self._cache.set(key, value)
        
        return value
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get int config value."""
        value = self.get(key)
        return int(value) if value is not None else default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float config value."""
        value = self.get(key)
        return float(value) if value is not None else default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get bool config value."""
        value = self.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes')


class MemoizationCache:
    """Function result memoization with TTL."""
    
    __slots__ = ('_cache', '_lock', '_ttl')
    
    def __init__(self, ttl: float = 3600):
        """Initialize memoization cache."""
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl
    
    def memoize(self, fn: Callable) -> Callable:
        """Decorator for function memoization."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import time
            
            # Create cache key from args and kwargs
            key = f"{fn.__name__}:{json.dumps([args, kwargs], default=str, sort_keys=True)}"
            
            with self._lock:
                if key in self._cache:
                    result, timestamp = self._cache[key]
                    if time.time() - timestamp < self._ttl:
                        return result
                    del self._cache[key]
            
            # Compute and cache result
            result = fn(*args, **kwargs)
            
            with self._lock:
                self._cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    
    def clear(self) -> None:
        """Clear memoization cache."""
        with self._lock:
            self._cache.clear()


class ConfigPrecomputation:
    """Pre-computes configuration-dependent values."""
    
    __slots__ = ('_precomputed', '_lock', '_config')
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize precomputation."""
        self._precomputed: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._config = config
    
    def precompute_all(self) -> None:
        """Pre-compute all values."""
        with self._lock:
            self._precomputed['model_params'] = self._compute_model_params()
            self._precomputed['device_config'] = self._compute_device_config()
            self._precomputed['optimization_flags'] = self._compute_optimization_flags()
    
    def _compute_model_params(self) -> Dict[str, Any]:
        """Compute model parameters."""
        return {
            'max_length': self._config.get('model', {}).get('max_length', 512),
            'vocab_size': self._config.get('model', {}).get('vocab_size', 50000),
        }
    
    def _compute_device_config(self) -> Dict[str, Any]:
        """Compute device configuration."""
        return {
            'device': self._config.get('device', 'cpu'),
            'cuda_enabled': self._config.get('use_cuda', False),
        }
    
    def _compute_optimization_flags(self) -> Dict[str, bool]:
        """Compute optimization flags."""
        return {
            'use_cache': self._config.get('cache_enabled', True),
            'use_batch': self._config.get('batch_processing', True),
        }
    
    def get_precomputed(self, key: str) -> Optional[Any]:
        """Get precomputed value."""
        with self._lock:
            return self._precomputed.get(key)


# Global instances
_config_cache = ConfigCache()
_memoization_cache = MemoizationCache()


def get_config_cache() -> ConfigCache:
    """Get global config cache."""
    return _config_cache


def get_memoization_cache() -> MemoizationCache:
    """Get global memoization cache."""
    return _memoization_cache


def memoize(fn: Callable) -> Callable:
    """Memoize a function."""
    return _memoization_cache.memoize(fn)
