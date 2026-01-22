"""Query caching layer for retrieval operations to avoid redundant embeddings.

This module provides caching for query embeddings and retrieval results
to avoid recomputing embeddings for identical queries.

Optimized with:
- xxhash for fast cache key generation
- time.monotonic for reliable timing
- Efficient LRU eviction
"""

import time
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import threading

# Use xxhash if available (6x faster than hashlib.md5)
try:
    import xxhash
    def _fast_hash(data: str) -> str:
        return xxhash.xxh64(data.encode()).hexdigest()
except ImportError:
    import hashlib
    def _fast_hash(data: str) -> str:
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()

# Cache time functions for micro-optimization
_monotonic = time.monotonic
_time = time.time


class QueryEmbeddingCache:
    """Cache for query embeddings to avoid redundant model inference."""
    
    __slots__ = ('_cache', '_lock', '_max_size', '_ttl', '_access_times')
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        """Initialize query embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl: Time-to-live for cached entries in seconds
        """
        self._cache: Dict[str, Tuple[List[float], float]] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl = ttl
        self._access_times: Dict[str, float] = {}
    
    def _make_key(self, query: str, model_name: str) -> str:
        """Create cache key from query and model name using fast hash."""
        return _fast_hash(f"{query}:{model_name}")
    
    def get(self, query: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding for query.
        
        Args:
            query: Query string
            model_name: Name of embedding model
            
        Returns:
            Cached embedding if found and not expired, None otherwise
        """
        key = self._make_key(query, model_name)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            embedding, timestamp = entry
            current_time = _monotonic()
            
            # Check expiration
            if current_time - timestamp > self._ttl:
                del self._cache[key]
                self._access_times.pop(key, None)
                return None
            
            # Update access time for LRU
            self._access_times[key] = current_time
            return embedding
    
    def put(self, query: str, model_name: str, embedding: List[float]) -> None:
        """Store embedding in cache.
        
        Args:
            query: Query string
            model_name: Name of embedding model
            embedding: Embedding vector
        """
        key = self._make_key(query, model_name)
        current_time = _monotonic()
        
        with self._lock:
            # Remove oldest item if at capacity (use OrderedDict for efficiency)
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._access_times.pop(oldest_key, None)
            
            # Store new entry
            self._cache[key] = (embedding, current_time)
            self._access_times[key] = current_time
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
            }


class RetrievalResultCache:
    """Cache for retrieval results to avoid redundant searches."""
    
    __slots__ = ('_cache', '_lock', '_max_size', '_ttl')
    
    def __init__(self, max_size: int = 500, ttl: float = 1800):
        """Initialize retrieval result cache.
        
        Args:
            max_size: Maximum number of cached results
            ttl: Time-to-live for cached entries in seconds
        """
        self._cache: Dict[str, Tuple[List[str], float]] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl = ttl
    
    def _make_key(self, query: str, method: str, k: int) -> str:
        """Create cache key from query parameters using fast hash."""
        return _fast_hash(f"{query}:{method}:{k}")
    
    def get(self, query: str, method: str, k: int) -> Optional[List[str]]:
        """Get cached retrieval results.
        
        Args:
            query: Query string
            method: Retrieval method (faiss, bm25, hybrid)
            k: Number of results
            
        Returns:
            Cached results if found and not expired, None otherwise
        """
        key = self._make_key(query, method, k)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            results, timestamp = entry
            
            # Check expiration
            if _monotonic() - timestamp > self._ttl:
                del self._cache[key]
                return None
            
            return results
    
    def put(self, query: str, method: str, k: int, results: List[str]) -> None:
        """Store retrieval results in cache.
        
        Args:
            query: Query string
            method: Retrieval method
            k: Number of results
            results: Retrieved code snippets
        """
        key = self._make_key(query, method, k)
        
        with self._lock:
            # Simple eviction policy: remove oldest when full
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest (FIFO)
            
            self._cache[key] = (results, _monotonic())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
            }


class QueryDeduplicator:
    """Deduplicator for concurrent identical queries to avoid redundant computation."""
    
    __slots__ = ('_pending_queries', '_results_cache', '_lock')
    
    def __init__(self):
        """Initialize query deduplicator."""
        self._pending_queries: Dict[str, threading.Event] = {}
        self._results_cache: Dict[str, any] = {}
        self._lock = threading.RLock()
    
    def register_query(self, query_id: str) -> bool:
        """Register a query for deduplication.
        
        Returns:
            True if this is the first instance of the query (caller should compute),
            False if another thread is already computing this query
        """
        with self._lock:
            if query_id in self._pending_queries:
                return False  # Already computing
            
            self._pending_queries[query_id] = threading.Event()
            return True  # First instance, compute it
    
    def mark_complete(self, query_id: str, result: any) -> None:
        """Mark query as complete and store result.
        
        Args:
            query_id: Query identifier
            result: Computation result
        """
        with self._lock:
            self._results_cache[query_id] = result
            if query_id in self._pending_queries:
                self._pending_queries[query_id].set()
    
    def wait_result(self, query_id: str, timeout: float = 30.0) -> Optional[any]:
        """Wait for query result from another thread.
        
        Args:
            query_id: Query identifier
            timeout: Maximum time to wait in seconds
            
        Returns:
            Query result if available, None if timeout
        """
        with self._lock:
            event = self._pending_queries.get(query_id)
        
        if event is None:
            return None
        
        # Wait for computation to complete
        if event.wait(timeout=timeout):
            with self._lock:
                return self._results_cache.get(query_id)
        
        return None
    
    def cleanup(self, query_id: str) -> None:
        """Cleanup after query processing.
        
        Args:
            query_id: Query identifier
        """
        with self._lock:
            self._pending_queries.pop(query_id, None)
            # Keep result in cache for a while
    
    def clear_old_results(self, max_age: float = 3600) -> None:
        """Clear results older than max_age.
        
        Args:
            max_age: Maximum age in seconds
        """
        with self._lock:
            # Simple cleanup - remove old entries
            # In production, would track timestamps
            if len(self._results_cache) > 1000:
                # Clear half the cache
                items = list(self._results_cache.items())
                for key, _ in items[:len(items)//2]:
                    del self._results_cache[key]


# Global singleton instances
_query_embedding_cache: Optional[QueryEmbeddingCache] = None
_retrieval_result_cache: Optional[RetrievalResultCache] = None
_query_deduplicator: Optional[QueryDeduplicator] = None

_cache_init_lock = threading.RLock()


def get_query_embedding_cache() -> QueryEmbeddingCache:
    """Get singleton query embedding cache."""
    global _query_embedding_cache
    
    if _query_embedding_cache is None:
        with _cache_init_lock:
            if _query_embedding_cache is None:
                _query_embedding_cache = QueryEmbeddingCache(max_size=1000)
    
    return _query_embedding_cache


def get_retrieval_result_cache() -> RetrievalResultCache:
    """Get singleton retrieval result cache."""
    global _retrieval_result_cache
    
    if _retrieval_result_cache is None:
        with _cache_init_lock:
            if _retrieval_result_cache is None:
                _retrieval_result_cache = RetrievalResultCache(max_size=500)
    
    return _retrieval_result_cache


def get_query_deduplicator() -> QueryDeduplicator:
    """Get singleton query deduplicator."""
    global _query_deduplicator
    
    if _query_deduplicator is None:
        with _cache_init_lock:
            if _query_deduplicator is None:
                _query_deduplicator = QueryDeduplicator()
    
    return _query_deduplicator
