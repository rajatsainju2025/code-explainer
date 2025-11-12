"""Eager initialization and attribute resolution for reduced lazy-loading overhead.

This module provides utilities for eager computation of expensive attributes
at initialization time rather than deferred lazy evaluation.
"""

import threading
from typing import Any, Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class EagerProperty:
    """Descriptor for properties eagerly computed at initialization."""
    
    name: str
    compute_fn: Callable[[], Any]
    value: Optional[Any] = None
    computed: bool = False
    compute_time: float = 0.0
    
    def compute(self) -> Any:
        """Compute the property value."""
        start = time.time()
        self.value = self.compute_fn()
        self.compute_time = time.time() - start
        self.computed = True
        return self.value


class EagerInitializer:
    """Initializes expensive attributes eagerly at startup."""
    
    __slots__ = ('_properties', '_lock', '_initialized')
    
    def __init__(self):
        """Initialize eager initializer."""
        self._properties: Dict[str, EagerProperty] = {}
        self._lock = threading.RLock()
        self._initialized = False
    
    def register_property(self, name: str, compute_fn: Callable) -> None:
        """Register a property for eager initialization.
        
        Args:
            name: Property name
            compute_fn: Callable that computes the property
        """
        with self._lock:
            self._properties[name] = EagerProperty(name=name, compute_fn=compute_fn)
    
    def initialize_all(self) -> Dict[str, Any]:
        """Eagerly compute all registered properties.
        
        Returns:
            Dictionary of computed properties and their compute times
        """
        results = {}
        
        with self._lock:
            for name, prop in self._properties.items():
                try:
                    prop.compute()
                    results[name] = {
                        'value': prop.value,
                        'compute_time': prop.compute_time,
                        'success': True
                    }
                except Exception as e:
                    results[name] = {
                        'error': str(e),
                        'compute_time': 0.0,
                        'success': False
                    }
            
            self._initialized = True
        
        return results
    
    def get_property(self, name: str) -> Optional[Any]:
        """Get a property value (must be initialized first).
        
        Args:
            name: Property name
            
        Returns:
            Property value if initialized, None otherwise
        """
        with self._lock:
            prop = self._properties.get(name)
            if prop and prop.computed:
                return prop.value
        
        return None
    
    def is_initialized(self) -> bool:
        """Check if all properties have been initialized."""
        with self._lock:
            return self._initialized and all(p.computed for p in self._properties.values())
    
    def get_initialization_stats(self) -> Dict[str, Any]:
        """Get initialization statistics."""
        with self._lock:
            total_time = sum(p.compute_time for p in self._properties.values())
            
            return {
                'total_properties': len(self._properties),
                'initialized_count': sum(1 for p in self._properties.values() if p.computed),
                'total_compute_time': total_time,
                'avg_compute_time': total_time / len(self._properties) if self._properties else 0,
                'slowest_properties': sorted(
                    [(name, prop.compute_time) for name, prop in self._properties.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }


class CachedAttributeResolver:
    """Resolves and caches frequently-accessed attributes."""
    
    __slots__ = ('_cache', '_lock', '_resolvers', '_ttl')
    
    def __init__(self, ttl: float = 3600):
        """Initialize cached attribute resolver.
        
        Args:
            ttl: Time-to-live for cached values
        """
        self._cache: Dict[Tuple[str, str], Tuple[Any, float]] = {}  # (obj_id, attr) -> (value, time)
        self._lock = threading.RLock()
        self._resolvers: Dict[str, Callable] = {}  # attr_name -> resolver_fn
        self._ttl = ttl
    
    def register_resolver(self, attr_name: str, resolver_fn: Callable) -> None:
        """Register a custom resolver for an attribute.
        
        Args:
            attr_name: Attribute name
            resolver_fn: Function that resolves the attribute
        """
        with self._lock:
            self._resolvers[attr_name] = resolver_fn
    
    def resolve_attribute(self, obj: Any, attr_name: str) -> Optional[Any]:
        """Resolve and cache an attribute value.
        
        Args:
            obj: Object to resolve attribute from
            attr_name: Attribute name
            
        Returns:
            Resolved attribute value
        """
        obj_id = id(obj)
        cache_key = (str(obj_id), attr_name)
        
        # Check cache
        with self._lock:
            if cache_key in self._cache:
                value, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._ttl:
                    return value
                else:
                    del self._cache[cache_key]
        
        # Resolve using registered resolver or getattr
        try:
            with self._lock:
                resolver = self._resolvers.get(attr_name)
            
            if resolver:
                value = resolver(obj)
            else:
                value = getattr(obj, attr_name)
            
            # Cache result
            with self._lock:
                self._cache[cache_key] = (value, time.time())
            
            return value
        except AttributeError:
            return None
    
    def invalidate_attribute(self, obj: Any, attr_name: str) -> None:
        """Invalidate cached attribute for an object."""
        obj_id = id(obj)
        cache_key = (str(obj_id), attr_name)
        
        with self._lock:
            self._cache.pop(cache_key, None)


class LazyToEagerMigration:
    """Utility to migrate lazy properties to eager initialization."""
    
    __slots__ = ('_lazy_props', '_eager_init', '_lock')
    
    def __init__(self):
        """Initialize migration utility."""
        self._lazy_props: Dict[str, Callable[[], Any]] = {}
        self._eager_init = EagerInitializer()
        self._lock = threading.RLock()
    
    def register_lazy_property(self, name: str, lazy_fn: Callable) -> None:
        """Register a lazy property for potential eager initialization.
        
        Args:
            name: Property name
            lazy_fn: Lazy property function
        """
        with self._lock:
            self._lazy_props[name] = lazy_fn
            # Also register for eager init
            self._eager_init.register_property(name, lazy_fn)
    
    def migrate_to_eager(self, properties: List[str]) -> Dict[str, Any]:
        """Migrate specific lazy properties to eager initialization.
        
        Args:
            properties: List of property names to eagerly initialize
            
        Returns:
            Results of eager initialization
        """
        # Initialize only specified properties
        results = {}
        
        with self._lock:
            for name in properties:
                if name in self._lazy_props:
                    prop = EagerProperty(name=name, compute_fn=self._lazy_props[name])
                    try:
                        prop.compute()
                        results[name] = {
                            'value': prop.value,
                            'compute_time': prop.compute_time
                        }
                    except Exception as e:
                        results[name] = {'error': str(e)}
        
        return results
    
    def get_initialization_candidates(self, access_threshold: int = 100) -> List[Tuple[str, int]]:
        """Identify lazy properties that are accessed frequently.
        
        Args:
            access_threshold: Access count threshold
            
        Returns:
            List of (property_name, access_count) tuples
        """
        # In real implementation, would track access counts
        # This is a placeholder
        return []


class StartupOptimizer:
    """Optimizes application startup by managing initialization order."""
    
    __slots__ = ('_phases', '_lock', '_current_phase')
    
    def __init__(self):
        """Initialize startup optimizer."""
        self._phases: Dict[int, List[Callable]] = {}  # phase -> [init functions]
        self._lock = threading.RLock()
        self._current_phase = -1
    
    def register_init(self, phase: int, init_fn: Callable) -> None:
        """Register an initialization function for a phase.
        
        Args:
            phase: Phase number (0=earliest, higher=later)
            init_fn: Initialization function
        """
        with self._lock:
            if phase not in self._phases:
                self._phases[phase] = []
            self._phases[phase].append(init_fn)
    
    def run_startup(self) -> Dict[int, List[Tuple[str, float, bool]]]:
        """Run all initialization phases in order.
        
        Returns:
            Dictionary mapping phases to (function_name, exec_time, success)
        """
        results: Dict[int, List[Tuple[str, float, bool]]] = {}
        
        with self._lock:
            for phase in sorted(self._phases.keys()):
                results[phase] = []
                self._current_phase = phase
                
                for init_fn in self._phases[phase]:
                    start = time.time()
                    try:
                        init_fn()
                        exec_time = time.time() - start
                        results[phase].append((init_fn.__name__, exec_time, True))
                    except Exception as e:
                        exec_time = time.time() - start
                        results[phase].append((init_fn.__name__, exec_time, False))
        
        return results
    
    def get_critical_path(self) -> List[str]:
        """Get initialization critical path (slowest phase sequence).
        
        Returns:
            List of phase names
        """
        # Simplified implementation
        return [f"phase_{p}" for p in sorted(self._phases.keys())]


# Global singleton instances
_eager_initializer: Optional[EagerInitializer] = None
_cached_resolver: Optional[CachedAttributeResolver] = None
_startup_optimizer: Optional[StartupOptimizer] = None

_init_lock = threading.RLock()


def get_eager_initializer() -> EagerInitializer:
    """Get singleton eager initializer."""
    global _eager_initializer
    
    if _eager_initializer is None:
        with _init_lock:
            if _eager_initializer is None:
                _eager_initializer = EagerInitializer()
    
    return _eager_initializer


def get_cached_resolver() -> CachedAttributeResolver:
    """Get singleton cached resolver."""
    global _cached_resolver
    
    if _cached_resolver is None:
        with _init_lock:
            if _cached_resolver is None:
                _cached_resolver = CachedAttributeResolver()
    
    return _cached_resolver


def get_startup_optimizer() -> StartupOptimizer:
    """Get singleton startup optimizer."""
    global _startup_optimizer
    
    if _startup_optimizer is None:
        with _init_lock:
            if _startup_optimizer is None:
                _startup_optimizer = StartupOptimizer()
    
    return _startup_optimizer
