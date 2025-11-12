"""Connection and resource pooling for reusable object management.

This module provides pooling infrastructure for connections, file handles,
and other expensive resources to reduce allocation overhead.
"""

import threading
from typing import Any, Callable, Dict, List, Optional, Type, Deque
from collections import deque
import time


class ObjectPool:
    """Generic object pool for reusable resource management."""
    
    __slots__ = ('_available', '_in_use', '_factory', '_max_size', '_lock', '_stats')
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 10):
        """Initialize object pool.
        
        Args:
            factory: Callable that creates new objects
            max_size: Maximum objects to keep in pool
        """
        self._available: Deque = deque()
        self._in_use: set = set()
        self._factory = factory
        self._max_size = max_size
        self._lock = threading.RLock()
        self._stats = {
            'created': 0,
            'acquired': 0,
            'released': 0,
            'reused': 0,
        }
    
    def acquire(self) -> Any:
        """Acquire an object from the pool.
        
        Returns:
            Object from pool or newly created if pool is empty
        """
        with self._lock:
            if self._available:
                obj = self._available.popleft()
                self._in_use.add(id(obj))
                self._stats['reused'] += 1
                self._stats['acquired'] += 1
                return obj
            
            # Create new object if under limit
            if len(self._in_use) < self._max_size:
                obj = self._factory()
                self._in_use.add(id(obj))
                self._stats['created'] += 1
                self._stats['acquired'] += 1
                return obj
            
            # Pool full, create anyway (caller responsible for cleanup)
            obj = self._factory()
            self._stats['created'] += 1
            self._stats['acquired'] += 1
            return obj
    
    def release(self, obj: Any) -> None:
        """Release an object back to the pool.
        
        Args:
            obj: Object to release
        """
        obj_id = id(obj)
        
        with self._lock:
            if obj_id in self._in_use:
                self._in_use.remove(obj_id)
            
            # Only add back if pool has space
            if len(self._available) < self._max_size:
                self._available.append(obj)
            
            self._stats['released'] += 1
    
    def clear(self) -> None:
        """Clear the pool."""
        with self._lock:
            self._available.clear()
            self._in_use.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                **self._stats,
                'available': len(self._available),
                'in_use': len(self._in_use),
            }


class ConnectionPool:
    """Pool for managing reusable connections."""
    
    __slots__ = ('_pools', '_lock', '_timeout', '_max_connections_per_type')
    
    def __init__(self, timeout: float = 30.0, max_connections_per_type: int = 10):
        """Initialize connection pool.
        
        Args:
            timeout: Connection timeout in seconds
            max_connections_per_type: Max connections per connection type
        """
        self._pools: Dict[str, ObjectPool] = {}
        self._lock = threading.RLock()
        self._timeout = timeout
        self._max_connections_per_type = max_connections_per_type
    
    def register_connection_type(self, conn_type: str, factory: Callable) -> None:
        """Register a connection type and its factory.
        
        Args:
            conn_type: Type name for the connection
            factory: Callable that creates connections
        """
        with self._lock:
            self._pools[conn_type] = ObjectPool(factory, max_size=self._max_connections_per_type)
    
    def get_connection(self, conn_type: str) -> Any:
        """Get a connection from the pool.
        
        Args:
            conn_type: Type of connection
            
        Returns:
            Connection object
        """
        with self._lock:
            pool = self._pools.get(conn_type)
        
        if pool is None:
            raise ValueError(f"Unknown connection type: {conn_type}")
        
        return pool.acquire()
    
    def release_connection(self, conn_type: str, connection: Any) -> None:
        """Release a connection back to the pool.
        
        Args:
            conn_type: Type of connection
            connection: Connection to release
        """
        with self._lock:
            pool = self._pools.get(conn_type)
        
        if pool is not None:
            pool.release(connection)
    
    def clear_type(self, conn_type: str) -> None:
        """Clear connections of a specific type."""
        with self._lock:
            pool = self._pools.get(conn_type)
            if pool:
                pool.clear()
    
    def clear_all(self) -> None:
        """Clear all pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all connection types."""
        with self._lock:
            return {
                conn_type: pool.get_stats()
                for conn_type, pool in self._pools.items()
            }


class ResourceContext:
    """Context manager for automatic resource acquisition and release."""
    
    __slots__ = ('_pool', '_resource_type', '_resource')
    
    def __init__(self, pool: ObjectPool, resource_type: str = 'resource'):
        """Initialize resource context.
        
        Args:
            pool: Object pool to use
            resource_type: Type of resource
        """
        self._pool = pool
        self._resource_type = resource_type
        self._resource: Optional[Any] = None
    
    def __enter__(self) -> Any:
        """Acquire resource on enter."""
        self._resource = self._pool.acquire()
        return self._resource
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release resource on exit."""
        if self._resource is not None:
            self._pool.release(self._resource)
    
    def __call__(self, fn: Callable) -> Callable:
        """Decorator for automatic resource management."""
        def wrapper(*args, **kwargs):
            with self:
                return fn(self._resource, *args, **kwargs)
        return wrapper


class BufferPool:
    """Pool for reusable byte buffers."""
    
    __slots__ = ('_buffers', '_lock', '_buffer_size', '_max_buffers')
    
    def __init__(self, buffer_size: int = 8192, max_buffers: int = 10):
        """Initialize buffer pool.
        
        Args:
            buffer_size: Size of each buffer
            max_buffers: Maximum buffers to pool
        """
        self._buffers: Deque = deque()
        self._lock = threading.RLock()
        self._buffer_size = buffer_size
        self._max_buffers = max_buffers
        
        # Pre-allocate some buffers
        for _ in range(max_buffers // 2):
            self._buffers.append(bytearray(buffer_size))
    
    def acquire(self) -> bytearray:
        """Acquire a buffer from the pool."""
        with self._lock:
            if self._buffers:
                buf = self._buffers.popleft()
                buf.clear()
                return buf
        
        return bytearray(self._buffer_size)
    
    def release(self, buf: bytearray) -> None:
        """Release a buffer back to the pool."""
        if len(buf) != self._buffer_size:
            return  # Wrong size, discard
        
        with self._lock:
            if len(self._buffers) < self._max_buffers:
                buf.clear()
                self._buffers.append(buf)
    
    def get_available_count(self) -> int:
        """Get number of available buffers."""
        with self._lock:
            return len(self._buffers)


class StatePool:
    """Pool for reusable state objects."""
    
    __slots__ = ('_states', '_lock', '_factory', '_reset_fn')
    
    def __init__(self, factory: Callable[[], Any], reset_fn: Callable[[Any], None],
                max_size: int = 32):
        """Initialize state pool.
        
        Args:
            factory: Callable that creates state objects
            reset_fn: Callable that resets state objects
            max_size: Maximum states to pool
        """
        self._states: Deque = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._factory = factory
        self._reset_fn = reset_fn
        
        # Pre-populate
        for _ in range(max_size // 2):
            self._states.append(factory())
    
    def acquire(self) -> Any:
        """Acquire a state object."""
        with self._lock:
            if self._states:
                state = self._states.popleft()
                self._reset_fn(state)
                return state
        
        return self._factory()
    
    def release(self, state: Any) -> None:
        """Release a state object."""
        with self._lock:
            if len(self._states) < self._states.maxlen:
                self._reset_fn(state)
                self._states.append(state)
    
    def get_available_count(self) -> int:
        """Get number of available states."""
        with self._lock:
            return len(self._states)


# Global singleton instances
_connection_pool: Optional[ConnectionPool] = None
_buffer_pool: Optional[BufferPool] = None

_pool_init_lock = threading.RLock()


def get_connection_pool() -> ConnectionPool:
    """Get singleton connection pool."""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_init_lock:
            if _connection_pool is None:
                _connection_pool = ConnectionPool()
    
    return _connection_pool


def get_buffer_pool() -> BufferPool:
    """Get singleton buffer pool."""
    global _buffer_pool
    
    if _buffer_pool is None:
        with _pool_init_lock:
            if _buffer_pool is None:
                _buffer_pool = BufferPool()
    
    return _buffer_pool
