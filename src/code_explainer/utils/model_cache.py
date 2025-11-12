"""Caching layer for frequently accessed model attributes.

This module reduces repeated property lookups and attribute access
overhead by providing cached access to model metadata.
"""

import threading
from typing import Any, Dict, Optional, Type
from dataclasses import dataclass
import time


@dataclass
class ModelMetadata:
    """Cached model metadata."""
    device: str
    dtype: str
    model_name: str
    architecture: str
    max_length: int
    vocabulary_size: int
    parameter_count: int
    timestamp: float
    
    def is_expired(self, ttl: float = 3600) -> bool:
        """Check if metadata is expired."""
        return time.time() - self.timestamp > ttl


class ModelAttributeCache:
    """Cache for frequently accessed model attributes."""
    
    __slots__ = ('_cache', '_lock', '_ttl', '_hits', '_misses')
    
    def __init__(self, ttl: float = 3600):
        """Initialize model attribute cache.
        
        Args:
            ttl: Time-to-live for cached attributes in seconds
        """
        self._cache: Dict[Tuple[int, str], Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl
        self._hits = 0
        self._misses = 0
    
    def get(self, obj_id: int, attr_name: str) -> Optional[Any]:
        """Get cached attribute value.
        
        Args:
            obj_id: Object ID (usually id(object))
            attr_name: Attribute name
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        key = (obj_id, attr_name)
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            value, timestamp = self._cache[key]
            
            # Check expiration
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            self._hits += 1
            return value
    
    def set(self, obj_id: int, attr_name: str, value: Any) -> None:
        """Cache an attribute value.
        
        Args:
            obj_id: Object ID (usually id(object))
            attr_name: Attribute name
            value: Value to cache
        """
        key = (obj_id, attr_name)
        
        with self._lock:
            self._cache[key] = (value, time.time())
    
    def invalidate(self, obj_id: int, attr_name: Optional[str] = None) -> None:
        """Invalidate cached values for an object.
        
        Args:
            obj_id: Object ID
            attr_name: Specific attribute to invalidate, or None for all
        """
        with self._lock:
            if attr_name is None:
                # Invalidate all attributes for this object
                keys_to_remove = [k for k in self._cache if k[0] == obj_id]
                for key in keys_to_remove:
                    del self._cache[key]
            else:
                key = (obj_id, attr_name)
                if key in self._cache:
                    del self._cache[key]
    
    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'cached_entries': len(self._cache),
            }


class CachedModelWrapper:
    """Wrapper that caches access to model attributes."""
    
    __slots__ = ('_model', '_metadata_cache', '_cache', '_lock')
    
    def __init__(self, model: Any):
        """Initialize cached model wrapper.
        
        Args:
            model: Model object to wrap
        """
        self._model = model
        self._metadata_cache: Optional[ModelMetadata] = None
        self._cache = ModelAttributeCache()
        self._lock = threading.RLock()
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute with caching."""
        # Try cache first
        obj_id = id(self._model)
        cached_value = self._cache.get(obj_id, name)
        if cached_value is not None:
            return cached_value
        
        # Get from model and cache
        try:
            value = getattr(self._model, name)
            self._cache.set(obj_id, name, value)
            return value
        except AttributeError:
            raise
    
    def refresh_metadata(self) -> ModelMetadata:
        """Refresh and cache model metadata."""
        with self._lock:
            try:
                metadata = ModelMetadata(
                    device=str(getattr(self._model, 'device', 'cpu')),
                    dtype=str(getattr(self._model, 'dtype', 'float32')),
                    model_name=str(getattr(self._model, 'model_name', 'unknown')),
                    architecture=str(getattr(self._model, 'architecture', 'unknown')),
                    max_length=int(getattr(self._model, 'max_length', 512)),
                    vocabulary_size=int(getattr(self._model, 'vocab_size', 0)),
                    parameter_count=int(getattr(self._model, 'num_parameters', 0)),
                    timestamp=time.time()
                )
                self._metadata_cache = metadata
                return metadata
            except Exception as e:
                # Return partial metadata on error
                return ModelMetadata(
                    device='unknown',
                    dtype='unknown',
                    model_name='unknown',
                    architecture='unknown',
                    max_length=512,
                    vocabulary_size=0,
                    parameter_count=0,
                    timestamp=time.time()
                )
    
    def get_metadata(self, refresh: bool = False) -> ModelMetadata:
        """Get cached model metadata.
        
        Args:
            refresh: Force refresh of metadata
            
        Returns:
            ModelMetadata object
        """
        with self._lock:
            if refresh or self._metadata_cache is None:
                return self.refresh_metadata()
            
            if self._metadata_cache.is_expired():
                return self.refresh_metadata()
            
            return self._metadata_cache
    
    def invalidate_cache(self, attr_name: Optional[str] = None) -> None:
        """Invalidate cached attributes.
        
        Args:
            attr_name: Specific attribute to invalidate, or None for all
        """
        obj_id = id(self._model)
        self._cache.invalidate(obj_id, attr_name)
        if attr_name is None:
            with self._lock:
                self._metadata_cache = None


class AttributeAccessOptimizer:
    """Optimizer for frequently accessed object attributes."""
    
    __slots__ = ('_attributes', '_lock', '_access_counts')
    
    def __init__(self):
        """Initialize attribute access optimizer."""
        self._attributes: Dict[Tuple[Type, str], Any] = {}
        self._lock = threading.RLock()
        self._access_counts: Dict[Tuple[Type, str], int] = {}
    
    def access(self, obj: Any, attr_name: str) -> Any:
        """Access attribute with optimization.
        
        Args:
            obj: Object to access
            attr_name: Attribute name
            
        Returns:
            Attribute value
        """
        obj_type = type(obj)
        key = (obj_type, attr_name)
        
        with self._lock:
            # Track access count
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
        
        # Get attribute normally (no caching here, just tracking)
        return getattr(obj, attr_name)
    
    def get_hot_attributes(self, threshold: int = 100) -> Dict[Type, list]:
        """Get attributes accessed more than threshold times.
        
        Args:
            threshold: Access count threshold
            
        Returns:
            Dictionary mapping types to list of hot attributes
        """
        with self._lock:
            hot_attrs: Dict[Type, list] = {}
            
            for (obj_type, attr_name), count in self._access_counts.items():
                if count >= threshold:
                    if obj_type not in hot_attrs:
                        hot_attrs[obj_type] = []
                    hot_attrs[obj_type].append((attr_name, count))
            
            return hot_attrs
    
    def reset_counts(self) -> None:
        """Reset access counts."""
        with self._lock:
            self._access_counts.clear()


# Global singleton instances
_model_attribute_cache: Optional[ModelAttributeCache] = None
_attribute_optimizer: Optional[AttributeAccessOptimizer] = None

_cache_init_lock = threading.RLock()


def get_model_attribute_cache() -> ModelAttributeCache:
    """Get singleton model attribute cache."""
    global _model_attribute_cache
    
    if _model_attribute_cache is None:
        with _cache_init_lock:
            if _model_attribute_cache is None:
                _model_attribute_cache = ModelAttributeCache()
    
    return _model_attribute_cache


def get_attribute_optimizer() -> AttributeAccessOptimizer:
    """Get singleton attribute access optimizer."""
    global _attribute_optimizer
    
    if _attribute_optimizer is None:
        with _cache_init_lock:
            if _attribute_optimizer is None:
                _attribute_optimizer = AttributeAccessOptimizer()
    
    return _attribute_optimizer


def wrap_model(model: Any) -> CachedModelWrapper:
    """Wrap a model for optimized attribute access.
    
    Args:
        model: Model object to wrap
        
    Returns:
        CachedModelWrapper instance
    """
    return CachedModelWrapper(model)
