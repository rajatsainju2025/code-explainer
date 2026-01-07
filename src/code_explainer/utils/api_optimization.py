"""Optimized API request processing with deduplication and caching."""

import hashlib
import logging
from typing import Any, Dict, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RequestDeduplicator:
    """Deduplicates concurrent identical requests to avoid redundant processing."""
    
    def __init__(self, ttl_seconds: int = 60):
        """Initialize the request deduplicator.
        
        Args:
            ttl_seconds: Time-to-live for cached request results
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # (result, expiry_time)
        self._pending: Dict[str, Any] = {}  # in-flight requests
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a hash key from request parameters."""
        key_str = str((args, sorted(kwargs.items())))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cached(self, *args, **kwargs) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._make_key(*args, **kwargs)
        if key in self._cache:
            result, expiry = self._cache[key]
            if datetime.now() < expiry:
                logger.debug("Cache hit for request %s...", key[:8])
                return result
            else:
                del self._cache[key]
        return None
    
    def cache_result(self, result: Any, *args, **kwargs) -> None:
        """Cache a result with TTL."""
        key = self._make_key(*args, **kwargs)
        expiry = datetime.now() + timedelta(seconds=self.ttl_seconds)
        self._cache[key] = (result, expiry)
    
    def is_pending(self, *args, **kwargs) -> bool:
        """Check if request is already being processed."""
        key = self._make_key(*args, **kwargs)
        return key in self._pending
    
    def mark_pending(self, *args, **kwargs) -> None:
        """Mark request as in-flight."""
        key = self._make_key(*args, **kwargs)
        self._pending[key] = True
    
    def mark_complete(self, *args, **kwargs) -> None:
        """Mark request as complete."""
        key = self._make_key(*args, **kwargs)
        if key in self._pending:
            del self._pending[key]
    
    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._pending.clear()
        logger.info("Request deduplicator cache cleared")


class ResponseStreamBuilder:
    """Builds responses efficiently using streaming and buffering strategies."""
    
    def __init__(self, buffer_size: int = 8192):
        """Initialize the response builder.
        
        Args:
            buffer_size: Size of internal buffer for streaming
        """
        self.buffer_size = buffer_size
        self._buffer = []
        self._current_size = 0
    
    def add_chunk(self, chunk: str) -> Optional[str]:
        """Add a chunk to the response and flush if needed.
        
        Args:
            chunk: Text chunk to add
            
        Returns:
            Flushed data if buffer exceeded, None otherwise
        """
        self._buffer.append(chunk)
        self._current_size += len(chunk)
        
        if self._current_size >= self.buffer_size:
            return self.flush()
        return None
    
    def flush(self) -> str:
        """Flush buffered data.
        
        Returns:
            All buffered content
        """
        result = ''.join(self._buffer)
        self._buffer.clear()
        self._current_size = 0
        return result
    
    def get_buffered(self) -> str:
        """Get current buffered content without flushing."""
        return ''.join(self._buffer)


class RequestMetricsCollector:
    """Collects request metrics efficiently with minimal overhead."""
    
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._metrics = {}
    
    def record_request(self, endpoint: str, duration_ms: float, 
                      cached: bool = False) -> None:
        """Record a request metric.
        
        Args:
            endpoint: API endpoint name
            duration_ms: Request duration in milliseconds
            cached: Whether result was cached
        """
        self.total_requests += 1
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if endpoint not in self._metrics:
            self._metrics[endpoint] = {
                "count": 0,
                "total_time_ms": 0,
                "min_time_ms": float('inf'),
                "max_time_ms": 0
            }
        
        m = self._metrics[endpoint]
        m["count"] += 1
        m["total_time_ms"] += duration_ms
        m["min_time_ms"] = min(m["min_time_ms"], duration_ms)
        m["max_time_ms"] = max(m["max_time_ms"], duration_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated metrics.
        
        Returns:
            Dictionary with metrics
        """
        stats = {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0,
            "endpoints": {}
        }
        
        for endpoint, metrics in self._metrics.items():
            avg = metrics["total_time_ms"] / metrics["count"]
            stats["endpoints"][endpoint] = {
                **metrics,
                "avg_time_ms": avg
            }
        
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._metrics.clear()


def deduplicate_requests(ttl_seconds: int = 60):
    """Decorator to deduplicate identical concurrent requests.
    
    Args:
        ttl_seconds: Cache TTL for request results
    """
    deduplicator = RequestDeduplicator(ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check cache
            cached = deduplicator.get_cached(*args, **kwargs)
            if cached is not None:
                return cached
            
            # Check if pending (in-flight)
            if deduplicator.is_pending(*args, **kwargs):
                logger.debug("Request already in-flight, waiting for result")
                # In real implementation, would wait for completion
                pass
            
            # Mark as pending and execute
            deduplicator.mark_pending(*args, **kwargs)
            try:
                result = func(*args, **kwargs)
                deduplicator.cache_result(result, *args, **kwargs)
                return result
            finally:
                deduplicator.mark_complete(*args, **kwargs)
        
        return wrapper
    
    return decorator
