"""Context and state management optimization utilities.

This module provides efficient context management for requests and async operations
to reduce overhead from repeated context creation and management.
"""

from typing import Any, Dict, Optional, List
import threading
from contextvars import ContextVar


class OptimizedContext:
    """Optimized context with minimal overhead."""
    
    __slots__ = ('_state', '_lock', '_parent')
    
    def __init__(self, parent: Optional['OptimizedContext'] = None):
        """Initialize optimized context."""
        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._parent = parent
    
    def set(self, key: str, value: Any) -> None:
        """Set context value."""
        with self._lock:
            self._state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        with self._lock:
            if key in self._state:
                return self._state[key]
        
        if self._parent:
            return self._parent.get(key, default)
        
        return default
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._state.clear()


class ContextPool:
    """Pool of reusable context objects."""
    
    __slots__ = ('_contexts', '_lock')
    
    def __init__(self, max_size: int = 32):
        """Initialize context pool."""
        from collections import deque
        self._contexts: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()
        
        # Pre-populate
        for _ in range(max_size // 2):
            self._contexts.append(OptimizedContext())
    
    def acquire(self) -> OptimizedContext:
        """Acquire context from pool."""
        with self._lock:
            if self._contexts:
                return self._contexts.popleft()
        return OptimizedContext()
    
    def release(self, context: OptimizedContext) -> None:
        """Release context back to pool."""
        context._state.clear()
        with self._lock:
            if len(self._contexts) < self._contexts.maxlen:
                self._contexts.append(context)


class ThreadSafeContext:
    """Thread-safe context using contextvars."""
    
    __slots__ = ('_context_var', '_default')
    
    def __init__(self, name: str, default: Any = None):
        """Initialize thread-safe context."""
        self._context_var = ContextVar(name, default=default)
        self._default = default
    
    def set(self, value: Any) -> None:
        """Set context value."""
        self._context_var.set(value)
    
    def get(self) -> Any:
        """Get context value."""
        return self._context_var.get(self._default)


class RequestContext:
    """Fast request context for API calls."""
    
    __slots__ = (
        'request_id', 'user_id', 'start_time', 'endpoint',
        'method', 'status_code', 'metadata'
    )
    
    def __init__(self):
        """Initialize request context."""
        self.request_id: str = ''
        self.user_id: str = ''
        self.start_time: float = 0.0
        self.endpoint: str = ''
        self.method: str = ''
        self.status_code: int = 0
        self.metadata: Dict[str, Any] = {}
    
    def reset(self) -> None:
        """Reset context."""
        self.request_id = ''
        self.user_id = ''
        self.start_time = 0.0
        self.endpoint = ''
        self.method = ''
        self.status_code = 0
        self.metadata.clear()


# Global instances
_context_pool = ContextPool()
_request_context = ContextVar('request_context', default=None)


def get_context() -> OptimizedContext:
    """Get optimized context."""
    return _context_pool.acquire()


def release_context(ctx: OptimizedContext) -> None:
    """Release context back to pool."""
    _context_pool.release(ctx)


def get_request_context() -> Optional[RequestContext]:
    """Get current request context."""
    return _request_context.get()


def set_request_context(ctx: RequestContext) -> None:
    """Set current request context."""
    _request_context.set(ctx)
