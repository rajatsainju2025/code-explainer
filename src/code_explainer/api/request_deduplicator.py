"""Request deduplication to avoid redundant processing of identical concurrent requests."""

import time
import threading
from typing import Any, Dict, Optional, Callable, Tuple
from collections import OrderedDict

try:
    import xxhash
    def fast_hash(data: bytes) -> str:
        return xxhash.xxh64(data).hexdigest()
except ImportError:
    import hashlib
    def fast_hash(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()


class RequestDeduplicator:
    """Deduplicates identical concurrent requests to avoid redundant computation."""
    
    __slots__ = ('_pending', '_results', '_lock', '_ttl', '_max_size')
    
    def __init__(self, ttl: float = 60.0, max_size: int = 1000):
        """Initialize request deduplicator.
        
        Args:
            ttl: Time-to-live for cached results in seconds
            max_size: Maximum number of cached results
        """
        self._pending: Dict[str, threading.Event] = {}
        self._results: Dict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._ttl = ttl
        self._max_size = max_size
    
    def _make_request_hash(self, endpoint: str, **kwargs) -> str:
        """Create hash of request parameters."""
        # Build canonical request representation
        parts = [endpoint]
        for key in sorted(kwargs.keys()):
            value = kwargs[key]
            if isinstance(value, (list, dict)):
                # For unhashable types, use JSON representation
                import orjson
                value_str = orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()
            else:
                value_str = str(value)
            parts.append(f"{key}={value_str}")
        
        return fast_hash('|'.join(parts).encode())
    
    def should_deduplicate(self, endpoint: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Check if request should be deduplicated.
        
        Returns:
            Tuple of (should_compute, request_id) where:
            - should_compute: True if this is first instance (caller should compute)
            - request_id: Unique ID for this request
        """
        request_id = self._make_request_hash(endpoint, **kwargs)
        
        with self._lock:
            # Check if result is cached and not expired
            if request_id in self._results:
                result, timestamp = self._results[request_id]
                if time.time() - timestamp < self._ttl:
                    # Cache hit - don't recompute
                    return (False, request_id)
                else:
                    # Cache expired - remove it
                    del self._results[request_id]
            
            # Check if computation is in progress
            if request_id in self._pending:
                # Another thread is computing this - wait for it
                return (False, request_id)
            
            # First instance - mark as pending
            self._pending[request_id] = threading.Event()
            return (True, request_id)
    
    def store_result(self, request_id: str, result: Any) -> None:
        """Store result of computation and signal pending requests.
        
        Args:
            request_id: Request identifier
            result: Computation result
        """
        with self._lock:
            # Store result
            self._results[request_id] = (result, time.time())
            
            # Evict oldest if at capacity
            if len(self._results) > self._max_size:
                self._results.popitem(last=False)
            
            # Signal pending requests
            if request_id in self._pending:
                self._pending[request_id].set()
    
    def wait_result(self, request_id: str, timeout: float = 30.0) -> Optional[Any]:
        """Wait for result from another thread.
        
        Args:
            request_id: Request identifier
            timeout: Maximum wait time in seconds
            
        Returns:
            Result if available, None if timeout
        """
        with self._lock:
            event = self._pending.get(request_id)
        
        if event is None:
            return None
        
        # Wait for computation
        if event.wait(timeout=timeout):
            with self._lock:
                if request_id in self._results:
                    result, _ = self._results[request_id]
                    return result
        
        return None
    
    def cleanup(self, request_id: str) -> None:
        """Cleanup after request processing.
        
        Args:
            request_id: Request identifier
        """
        with self._lock:
            self._pending.pop(request_id, None)
    
    def clear_expired(self) -> int:
        """Clear expired cached results.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        removed = 0
        
        with self._lock:
            expired_keys = [
                key for key, (_, timestamp) in self._results.items()
                if current_time - timestamp >= self._ttl
            ]
            
            for key in expired_keys:
                del self._results[key]
                removed += 1
        
        return removed
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplicator statistics."""
        with self._lock:
            return {
                'pending_requests': len(self._pending),
                'cached_results': len(self._results),
                'max_size': self._max_size,
            }


class CodeExplanationDeduplicator(RequestDeduplicator):
    """Specialized deduplicator for code explanation requests."""
    
    def __init__(self, ttl: float = 300.0, max_size: int = 500):
        """Initialize code explanation deduplicator.
        
        Args:
            ttl: Cache TTL (default 5 minutes for explanations)
            max_size: Maximum cached explanations
        """
        super().__init__(ttl=ttl, max_size=max_size)
    
    def should_deduplicate_explanation(self, code: str, strategy: str = 'vanilla',
                                       max_length: int = 512) -> Tuple[bool, Optional[str]]:
        """Check if explanation request should be deduplicated.
        
        Args:
            code: Source code to explain
            strategy: Explanation strategy
            max_length: Maximum explanation length
            
        Returns:
            Tuple of (should_compute, request_id)
        """
        return self.should_deduplicate(
            endpoint='explain',
            code_hash=fast_hash(code.encode()),  # Use faster hash
            strategy=strategy,
            max_length=max_length
        )


class BatchRequestDeduplicator(RequestDeduplicator):
    """Specialized deduplicator for batch requests."""
    
    def __init__(self, ttl: float = 60.0, max_size: int = 100):
        """Initialize batch request deduplicator.
        
        Args:
            ttl: Cache TTL (default 1 minute for batches)
            max_size: Maximum cached batch results
        """
        super().__init__(ttl=ttl, max_size=max_size)
    
    def should_deduplicate_batch(self, codes: list, strategy: str = 'vanilla',
                                max_length: int = 512) -> Tuple[bool, Optional[str]]:
        """Check if batch request should be deduplicated.
        
        Args:
            codes: List of codes to explain
            strategy: Explanation strategy
            max_length: Maximum explanation length
            
        Returns:
            Tuple of (should_compute, request_id)
        """
        import orjson
        # Create hash of codes list for uniqueness
        codes_hash = fast_hash(
            orjson.dumps(codes, option=orjson.OPT_SORT_KEYS)
        )
        
        return self.should_deduplicate(
            endpoint='explain/batch',
            codes_hash=codes_hash,
            strategy=strategy,
            max_length=max_length
        )


# Global singleton instances
_request_deduplicator: Optional[RequestDeduplicator] = None
_code_explanation_deduplicator: Optional[CodeExplanationDeduplicator] = None
_batch_request_deduplicator: Optional[BatchRequestDeduplicator] = None

_dedup_init_lock = threading.RLock()


def get_request_deduplicator() -> RequestDeduplicator:
    """Get singleton request deduplicator."""
    global _request_deduplicator
    
    if _request_deduplicator is None:
        with _dedup_init_lock:
            if _request_deduplicator is None:
                _request_deduplicator = RequestDeduplicator()
    
    return _request_deduplicator


def get_code_explanation_deduplicator() -> CodeExplanationDeduplicator:
    """Get singleton code explanation deduplicator."""
    global _code_explanation_deduplicator
    
    if _code_explanation_deduplicator is None:
        with _dedup_init_lock:
            if _code_explanation_deduplicator is None:
                _code_explanation_deduplicator = CodeExplanationDeduplicator()
    
    return _code_explanation_deduplicator


def get_batch_request_deduplicator() -> BatchRequestDeduplicator:
    """Get singleton batch request deduplicator."""
    global _batch_request_deduplicator
    
    if _batch_request_deduplicator is None:
        with _dedup_init_lock:
            if _batch_request_deduplicator is None:
                _batch_request_deduplicator = BatchRequestDeduplicator()
    
    return _batch_request_deduplicator
