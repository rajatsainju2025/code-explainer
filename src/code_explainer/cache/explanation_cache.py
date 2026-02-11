"""Explanation cache implementation."""

import heapq
import time
import threading
from typing import Any, Dict, Optional

# Use orjson for faster JSON operations if available
try:
    import orjson
    def json_loads(s): return orjson.loads(s)
    def json_dumps(obj): return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode()
except ImportError:
    import json
    json_loads = json.loads
    def json_dumps(obj): return json.dumps(obj, separators=(',', ':'))

from .base_cache import BaseCache, MemoryCache
from .models import CacheConfig, CacheStats
from .utils import (calculate_cache_score, compress_data, decompress_data,
                   generate_cache_key, is_expired, safe_file_operation)


class ExplanationCache(BaseCache):
    """Cache for code explanations to avoid redundant model calls.
    
    Features:
    - Write-behind batching: queues index updates and flushes periodically
    - Reduces disk I/O for high-frequency cache writes
    - Maintains consistency with occasional forced flushes
    """

    def __init__(
        self,
        cache_dir: str = ".cache/explanations",
        max_size: int = 1000,
        ttl_seconds: int = 86400,  # 24 hours
        compression_enabled: bool = True,
        memory_cache_size: int = 100,
        write_behind_batch_size: int = 10,
        write_behind_flush_interval: float = 5.0
    ):
        config = CacheConfig(
            cache_dir=cache_dir,
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            compression_enabled=compression_enabled,
            memory_cache_size=memory_cache_size
        )
        super().__init__(config)
        # expose max_size for test assertions
        self.max_size = max_size

        self._index_file = self.cache_dir / "index.json"
        self._memory_cache = MemoryCache(self.config.memory_cache_size)
        self._index = self._load_index()
        
        # Write-behind batching configuration
        self.write_behind_batch_size = write_behind_batch_size
        self.write_behind_flush_interval = write_behind_flush_interval
        self._pending_writes_queue: deque = deque()
        self._last_flush_time = time.monotonic()  # Use monotonic for intervals
        self._write_behind_lock = threading.Lock()
        
        # Cache hit/miss tracking
        self._hits = 0
        self._misses = 0

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index."""
        data = safe_file_operation("load", self._index_file, "r")
        if data:
            try:
                return json_loads(data)
            except (ValueError, TypeError):
                pass
        return {}

    def _should_flush_writes(self) -> bool:
        """Check if pending writes should be flushed.
        
        Returns True if:
        - Batch size threshold reached, OR
        - Time interval elapsed since last flush
        """
        queue_len = len(self._pending_writes_queue)
        if queue_len >= self.write_behind_batch_size:
            return True
        
        time_elapsed = time.monotonic() - self._last_flush_time
        return time_elapsed >= self.write_behind_flush_interval

    def _queue_index_write(self, cache_key: str, operation: str = "update") -> None:
        """Queue an index write operation for batched flushing.
        
        Args:
            cache_key: Key of cache entry being modified
            operation: Type of operation (update, delete, cleanup)
        """
        with self._write_behind_lock:
            self._pending_writes_queue.append({
                "key": cache_key,
                "operation": operation,
            })
            
            # Check if we should flush now
            if self._should_flush_writes():
                self._flush_pending_writes_unsafe()

    def _flush_pending_writes_unsafe(self) -> None:
        """Internal flush without lock (must be called with _write_behind_lock held)."""
        if not self._pending_writes_queue:
            return
        
        # Save index once for all pending writes
        self._save_index(force=True)
        
        # Clear pending writes queue
        self._pending_writes_queue.clear()
        self._last_flush_time = time.monotonic()

    def _save_index(self, force: bool = False) -> None:
        """Save the cache index.
        
        Args:
            force: Force immediate save, even if no pending writes queued
        """
        # Use compact JSON for smaller file size
        data = json_dumps(self._index)
        safe_file_operation("save", self._index_file, "w", data)

    def get(self, code: str, strategy: str, model_name: str) -> Optional[str]:
        """Get a cached explanation if available."""
        cache_key = generate_cache_key(code, strategy, model_name)

        with self._lock:
            # Check memory cache first (fastest path)
            memory_result = self._memory_cache.get(cache_key)
            if memory_result is not None:
                entry = memory_result
                if not is_expired(entry['timestamp'], self.config.ttl_seconds):
                    self._hits += 1
                    return entry['data']
                else:
                    self._memory_cache.put(cache_key, None)  # Mark as expired

            # Check disk cache
            entry = self._index.get(cache_key)
            if entry is None:
                self._misses += 1
                return None

            if is_expired(entry.get('timestamp', 0), self.config.ttl_seconds):
                self._remove_entry(cache_key)
                self._misses += 1
                return None

            cache_file = self.cache_dir / f"{cache_key}.txt"
            compressed_data = safe_file_operation("read", cache_file, "rb")
            if compressed_data is None:
                # Remove stale index entry
                del self._index[cache_key]
                self._queue_index_write(cache_key, "delete")
                self._misses += 1
                return None

            try:
                explanation = decompress_data(compressed_data)

                # Update access metadata - queue for batched save
                entry["access_count"] = entry.get("access_count", 0) + 1
                entry["last_access"] = time.time()
                self._queue_index_write(cache_key, "update")

                # Add to memory cache
                self._memory_cache.put(cache_key, {
                    'data': explanation,
                    'timestamp': entry['timestamp']
                })

                self._hits += 1
                return explanation
            except (OSError, ValueError, KeyError):
                # Handle decompression or data corruption errors
                self._misses += 1
                return None
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate for monitoring."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def put(self, code: str, strategy: str, model_name: str, explanation: str) -> None:
        """Cache an explanation."""
        cache_key = generate_cache_key(code, strategy, model_name)
        cache_file = self.cache_dir / f"{cache_key}.txt"

        with self._lock:
            try:
                # Compress and write explanation to file
                compressed_data = compress_data(explanation)
                safe_file_operation("write", cache_file, "wb", compressed_data)

                # Update index
                current_time = time.time()
                self._index[cache_key] = {
                    "access_count": 1,
                    "code_length": len(code),
                    "strategy": strategy,
                    "model_name": model_name,
                    "timestamp": current_time,
                    "last_access": current_time,
                    "compressed": self.config.compression_enabled and len(explanation) >= 1000
                }

                # Add to memory cache
                self._memory_cache.put(cache_key, {
                    'data': explanation,
                    'timestamp': current_time
                })

                # Cleanup if cache is too large (async-friendly)
                if len(self._index) > self.config.max_size:
                    self._schedule_cleanup()
                
                # Queue index write for batched flushing
                self._queue_index_write(cache_key, "update")

            except Exception:
                pass  # Silent failure for caching
    
    def _schedule_cleanup(self) -> None:
        """Schedule cleanup without blocking the main operation."""
        # Mark cleanup needed, actual cleanup happens on next flush or explicit call
        self._cleanup_pending = True

    def flush(self) -> None:
        """Flush all pending index updates to disk."""
        with self._lock:
            with self._write_behind_lock:
                self._flush_pending_writes_unsafe()
                
                # Perform deferred cleanup if needed
                if getattr(self, '_cleanup_pending', False):
                    self._cleanup_if_needed()
                    self._cleanup_pending = False

    def _remove_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        try:
            cache_file.unlink(missing_ok=True)
        except (OSError, PermissionError):
            pass

        self._index.pop(cache_key, None)
        self._memory_cache.put(cache_key, None)  # Mark as removed

    def _cleanup_if_needed(self) -> None:
        """Remove least recently used entries if cache is too large.
        
        Uses heapq.nsmallest for O(n log k) efficiency.
        """
        index_size = len(self._index)
        if index_size <= self.config.max_size:
            return

        # Calculate how many to remove (with buffer to avoid frequent cleanups)
        num_to_remove = index_size - int(self.config.max_size * 0.9)  # Keep 90% capacity
        if num_to_remove <= 0:
            return
        
        current_time = time.time()
        
        # Use heapq.nsmallest for O(n log k) instead of full sort O(n log n)
        scored_entries = [
            (calculate_cache_score(entry, current_time), key)
            for key, entry in self._index.items()
        ]
        
        # Get only the entries we need to remove
        to_remove = heapq.nsmallest(num_to_remove, scored_entries)
        
        # Batch file deletions
        for _, key in to_remove:
            cache_file = self.cache_dir / f"{key}.txt"
            try:
                cache_file.unlink(missing_ok=True)
            except (OSError, PermissionError):
                pass
            self._index.pop(key, None)
        
        # Single index save after all removals
        self._save_index(force=True)

    def clear(self) -> None:
        """Clear all cached explanations."""
        with self._lock:
            try:
                for cache_file in self.cache_dir.glob("*.txt"):
                    try:
                        cache_file.unlink(missing_ok=True)
                    except (OSError, PermissionError):
                        pass
                self._index.clear()
                self._memory_cache.clear()
                self._pending_writes_queue.clear()
                self._save_index(force=True)
            except Exception:
                pass

    def size(self) -> int:
        """Get the number of cached explanations."""
        return len(self._index)

    def stats(self) -> CacheStats:
        """Get cache statistics with hit rate."""
        if not self._index:
            return CacheStats()

        # Efficient single-pass statistics gathering
        total_access = 0
        strategies_set = set()
        models_set = set()
        
        for entry in self._index.values():
            total_access += entry.get("access_count", 0)
            strategies_set.add(entry.get("strategy", ""))
            models_set.add(entry.get("model_name", ""))

        return CacheStats(
            size=len(self._index),
            total_access_count=total_access,
            avg_access_count=total_access / len(self._index) if self._index else 0,
            strategies=list(strategies_set),
            models=list(models_set),
            hit_rate=self.get_hit_rate()
        )