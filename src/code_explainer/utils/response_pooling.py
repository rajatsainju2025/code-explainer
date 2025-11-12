"""Response object pooling to reduce GC pressure from repeated allocations.

This module provides pooled response builders and reusable response objects
to minimize memory allocation overhead during request handling.
"""

import time
from typing import Any, Dict, Optional, List
from collections import deque
import threading


class ResponseBuilder:
    """Reusable builder for constructing API responses with minimal allocations."""
    
    __slots__ = ('_data', '_built')
    
    def __init__(self):
        """Initialize response builder."""
        self._data: Dict[str, Any] = {}
        self._built = False
    
    def set_field(self, key: str, value: Any) -> 'ResponseBuilder':
        """Set a field in the response."""
        self._data[key] = value
        return self
    
    def set_fields(self, **fields) -> 'ResponseBuilder':
        """Set multiple fields at once."""
        self._data.update(fields)
        return self
    
    def get_dict(self) -> Dict[str, Any]:
        """Get the response as a dictionary."""
        return self._data
    
    def reset(self) -> 'ResponseBuilder':
        """Reset builder for reuse."""
        self._data.clear()
        self._built = False
        return self


class ResponsePool:
    """Pool of reusable ResponseBuilder objects to reduce allocations."""
    
    __slots__ = ('_pool', '_lock', '_max_size')
    
    def __init__(self, max_size: int = 16):
        """Initialize response pool.
        
        Args:
            max_size: Maximum number of pooled builders to maintain
        """
        self._pool: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._max_size = max_size
        
        # Pre-populate pool
        for _ in range(max_size):
            self._pool.append(ResponseBuilder())
    
    def acquire(self) -> ResponseBuilder:
        """Acquire a ResponseBuilder from the pool."""
        with self._lock:
            if self._pool:
                return self._pool.popleft()
        return ResponseBuilder()
    
    def release(self, builder: ResponseBuilder) -> None:
        """Return a ResponseBuilder to the pool after resetting it."""
        builder.reset()
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(builder)
    
    def clear(self) -> None:
        """Clear the pool."""
        with self._lock:
            self._pool.clear()


class CachedMetricsDict:
    """Cached dictionary builder for frequently-requested metric structures.
    
    Avoids repeated dict construction for common metric patterns.
    """
    
    __slots__ = ('_templates', '_lock')
    
    def __init__(self):
        """Initialize cached metrics dict."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def get_template(self, template_name: str, 
                     builder_fn: callable) -> Dict[str, Any]:
        """Get or build a cached template."""
        if template_name not in self._templates:
            with self._lock:
                if template_name not in self._templates:
                    self._templates[template_name] = builder_fn()
        
        # Return a shallow copy to allow mutations without affecting cache
        return self._templates[template_name].copy()
    
    def clear(self) -> None:
        """Clear all cached templates."""
        with self._lock:
            self._templates.clear()


class RequestState:
    """Reusable request state object to hold request metadata.
    
    Uses __slots__ for memory efficiency.
    """
    
    __slots__ = (
        'request_id', 'start_time', 'method', 'path',
        'client_host', 'request_size', 'metadata'
    )
    
    def __init__(self):
        """Initialize request state."""
        self.request_id: str = ''
        self.start_time: float = 0.0
        self.method: str = ''
        self.path: str = ''
        self.client_host: str = 'unknown'
        self.request_size: int = 0
        self.metadata: Dict[str, Any] = {}
    
    def reset(self) -> None:
        """Reset state for reuse."""
        self.request_id = ''
        self.start_time = 0.0
        self.method = ''
        self.path = ''
        self.client_host = 'unknown'
        self.request_size = 0
        self.metadata.clear()
    
    def set_from_request(self, request_id: str, method: str, path: str,
                        client_host: str = 'unknown') -> 'RequestState':
        """Populate state from request data."""
        self.request_id = request_id
        self.start_time = time.time()
        self.method = method
        self.path = path
        self.client_host = client_host
        return self


class RequestStatePool:
    """Pool of reusable RequestState objects."""
    
    __slots__ = ('_pool', '_lock', '_max_size')
    
    def __init__(self, max_size: int = 32):
        """Initialize request state pool.
        
        Args:
            max_size: Maximum number of pooled states to maintain
        """
        self._pool: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._max_size = max_size
        
        # Pre-populate pool
        for _ in range(max_size):
            self._pool.append(RequestState())
    
    def acquire(self) -> RequestState:
        """Acquire a RequestState from the pool."""
        with self._lock:
            if self._pool:
                return self._pool.popleft()
        return RequestState()
    
    def release(self, state: RequestState) -> None:
        """Return a RequestState to the pool after resetting it."""
        state.reset()
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(state)


# Global pools
_response_pool = ResponsePool(max_size=16)
_request_state_pool = RequestStatePool(max_size=32)


def acquire_response_builder() -> ResponseBuilder:
    """Acquire a response builder from the global pool."""
    return _response_pool.acquire()


def release_response_builder(builder: ResponseBuilder) -> None:
    """Return a response builder to the global pool."""
    _response_pool.release(builder)


def acquire_request_state() -> RequestState:
    """Acquire a request state from the global pool."""
    return _request_state_pool.acquire()


def release_request_state(state: RequestState) -> None:
    """Return a request state to the global pool."""
    _request_state_pool.release(state)


def context_response_builder(fn):
    """Decorator to automatically pool response builders."""
    def wrapper(*args, **kwargs):
        builder = acquire_response_builder()
        try:
            return fn(builder, *args, **kwargs)
        finally:
            release_response_builder(builder)
    return wrapper


def context_request_state(fn):
    """Decorator to automatically pool request states."""
    def wrapper(*args, **kwargs):
        state = acquire_request_state()
        try:
            return fn(state, *args, **kwargs)
        finally:
            release_request_state(state)
    return wrapper
