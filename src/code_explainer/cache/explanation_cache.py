"""Explanation cache implementation."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from .base_cache import BaseCache, MemoryCache
from .models import CacheConfig, CacheStats, ExplanationEntry
from .utils import (calculate_cache_score, compress_data, decompress_data,
                   generate_cache_key, is_expired, safe_file_operation)


class ExplanationCache(BaseCache):
    """Cache for code explanations to avoid redundant model calls."""

    def __init__(
        self,
        cache_dir: str = ".cache/explanations",
        max_size: int = 1000,
        ttl_seconds: int = 86400,  # 24 hours
        compression_enabled: bool = True,
        memory_cache_size: int = 100
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
        self._pending_index_updates = False  # Track if index needs saving

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index."""
        data = safe_file_operation("load", self._index_file, "r")
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                pass
        return {}

    def _save_index(self, force: bool = False) -> None:
        """Save the cache index.
        
        Args:
            force: Force immediate save, otherwise batch updates
        """
        if not force and not self._pending_index_updates:
            return
        
        data = json.dumps(self._index, indent=2)
        safe_file_operation("save", self._index_file, "w", data)
        self._pending_index_updates = False

    def get(self, code: str, strategy: str, model_name: str) -> Optional[str]:
        """Get a cached explanation if available."""
        cache_key = generate_cache_key(code, strategy, model_name)

        with self._lock:
            # Check memory cache first
            memory_result = self._memory_cache.get(cache_key)
            if memory_result is not None:
                entry = memory_result
                if not is_expired(entry['timestamp'], self.config.ttl_seconds):
                    return entry['data']
                else:
                    self._memory_cache.put(cache_key, None)  # Mark as expired

            # Check disk cache
            if cache_key not in self._index:
                return None

            entry = self._index[cache_key]
            if is_expired(entry.get('timestamp', 0), self.config.ttl_seconds):
                self._remove_entry(cache_key)
                return None

            cache_file = self.cache_dir / f"{cache_key}.txt"
            compressed_data = safe_file_operation("read", cache_file, "rb")
            if compressed_data is None:
                # Remove stale index entry
                del self._index[cache_key]
                self._pending_index_updates = True
                return None

            try:
                explanation = decompress_data(compressed_data)

                # Update access metadata - batch these updates
                entry["access_count"] = entry.get("access_count", 0) + 1
                entry["last_access"] = time.time()
                self._pending_index_updates = True

                # Add to memory cache
                self._memory_cache.put(cache_key, {
                    'data': explanation,
                    'timestamp': entry['timestamp']
                })

                return explanation
            except Exception:
                return None

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

                # Cleanup if cache is too large
                self._cleanup_if_needed()
                
                # Batch save index after all operations
                self._save_index(force=True)

            except Exception:
                pass  # Silent failure for caching

    def flush(self) -> None:
        """Flush pending index updates to disk."""
        with self._lock:
            self._save_index(force=True)

    def _remove_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        try:
            cache_file.unlink(missing_ok=True)
        except Exception:
            pass

        if cache_key in self._index:
            del self._index[cache_key]

        self._memory_cache.put(cache_key, None)  # Mark as removed

    def _cleanup_if_needed(self) -> None:
        """Remove least recently used entries if cache is too large."""
        if len(self._index) <= self.config.max_size:
            return

        # Optimize: Calculate scores once and use heap for better performance
        current_time = time.time()
        entries_with_scores = []
        
        for key, entry in self._index.items():
            score = calculate_cache_score(entry, current_time)
            entries_with_scores.append((score, key))
        
        # Sort only once
        entries_with_scores.sort()

        # Batch remove entries and update index once at the end
        num_to_remove = len(self._index) - self.config.max_size + 1
        to_remove = entries_with_scores[:num_to_remove]
        
        for _, key in to_remove:
            cache_file = self.cache_dir / f"{key}.txt"
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass
            self._index.pop(key, None)
        
        # Single index save at the end instead of multiple saves
        self._save_index()

    def clear(self) -> None:
        """Clear all cached explanations."""
        try:
            for cache_file in self.cache_dir.glob("*.txt"):
                cache_file.unlink(missing_ok=True)
            self._index.clear()
            self._memory_cache.clear()
            self._save_index()
        except Exception:
            pass

    def size(self) -> int:
        """Get the number of cached explanations."""
        return len(self._index)

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        if not self._index:
            return CacheStats()

        total_access = sum(entry.get("access_count", 0) for entry in self._index.values())
        strategies = list(set(entry.get("strategy", "") for entry in self._index.values()))
        models = list(set(entry.get("model_name", "") for entry in self._index.values()))

        return CacheStats(
            size=len(self._index),
            total_access_count=total_access,
            avg_access_count=total_access / len(self._index) if self._index else 0,
            strategies=strategies,
            models=models
        )