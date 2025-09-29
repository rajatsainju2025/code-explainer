"""Caching utilities for code explanations and embeddings."""

import hashlib
import json
import logging
import pickle
import time
import gzip
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
from collections import OrderedDict
import heapq

logger = logging.getLogger(__name__)


class ExplanationCache:
    """Cache for code explanations to avoid redundant model calls."""

    def __init__(
        self,
        cache_dir: str = ".cache/explanations",
        max_size: int = 1000,
        ttl_seconds: int = 86400,  # 24 hours
        compression_enabled: bool = True,
        memory_cache_size: int = 100
    ):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached explanations
            ttl_seconds: Time-to-live for cache entries in seconds
            compression_enabled: Whether to compress large explanations
            memory_cache_size: Size of in-memory LRU cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.compression_enabled = compression_enabled
        self.memory_cache_size = memory_cache_size

        self._index_file = self.cache_dir / "index.json"
        self._lock = threading.RLock()  # Thread-safe operations
        self._memory_cache = OrderedDict()  # LRU memory cache
        self._load_index()

    def _load_index(self) -> None:
        """Load the cache index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save the cache index."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def _get_cache_key(self, code: str, strategy: str, model_name: str) -> str:
        """Generate a cache key for the given parameters."""
        content = f"{code}|{strategy}|{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is expired."""
        if 'timestamp' not in entry:
            return False
        return time.time() - entry['timestamp'] > self.ttl_seconds

    def _compress_data(self, data: str) -> bytes:
        """Compress data if enabled and beneficial."""
        if not self.compression_enabled or len(data) < 1000:
            return data.encode('utf-8')
        return gzip.compress(data.encode('utf-8'))

    def _decompress_data(self, data: bytes) -> str:
        """Decompress data if it was compressed."""
        try:
            decompressed = gzip.decompress(data)
            return decompressed.decode('utf-8')
        except gzip.BadGzipFile:
            return data.decode('utf-8')

    def get(self, code: str, strategy: str, model_name: str) -> Optional[str]:
        """Get a cached explanation if available.

        Args:
            code: The code to explain
            strategy: The prompting strategy used
            model_name: The model name used

        Returns:
            Cached explanation if available, None otherwise
        """
        cache_key = self._get_cache_key(code, strategy, model_name)

        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                if not self._is_expired(entry):
                    # Move to end (most recently used)
                    self._memory_cache.move_to_end(cache_key)
                    return entry['data']
                else:
                    del self._memory_cache[cache_key]

            # Check disk cache
            if cache_key not in self._index:
                return None

            entry = self._index[cache_key]
            if self._is_expired(entry):
                # Remove expired entry
                self._remove_entry(cache_key)
                return None

            cache_file = self.cache_dir / f"{cache_key}.txt"
            if not cache_file.exists():
                # Remove stale index entry
                del self._index[cache_key]
                self._save_index()
                return None

            try:
                with open(cache_file, "rb") as f:
                    compressed_data = f.read()
                    explanation = self._decompress_data(compressed_data)

                # Update access metadata
                entry["access_count"] += 1
                entry["last_access"] = time.time()
                self._save_index()

                # Add to memory cache
                self._add_to_memory_cache(cache_key, explanation)

                return explanation
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")
                return None

    def _add_to_memory_cache(self, key: str, data: str) -> None:
        """Add entry to memory cache with LRU eviction."""
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        else:
            if len(self._memory_cache) >= self.memory_cache_size:
                # Remove least recently used
                self._memory_cache.popitem(last=False)
            self._memory_cache[key] = {
                'data': data,
                'timestamp': time.time()
            }

    def put(self, code: str, strategy: str, model_name: str, explanation: str) -> None:
        """Cache an explanation.

        Args:
            code: The code that was explained
            strategy: The prompting strategy used
            model_name: The model name used
            explanation: The explanation to cache
        """
        cache_key = self._get_cache_key(code, strategy, model_name)
        cache_file = self.cache_dir / f"{cache_key}.txt"

        with self._lock:
            try:
                # Compress and write explanation to file
                compressed_data = self._compress_data(explanation)
                with open(cache_file, "wb") as f:
                    f.write(compressed_data)

                # Update index
                current_time = time.time()
                self._index[cache_key] = {
                    "access_count": 1,
                    "code_length": len(code),
                    "strategy": strategy,
                    "model_name": model_name,
                    "timestamp": current_time,
                    "last_access": current_time,
                    "compressed": self.compression_enabled and len(explanation) >= 1000
                }

                # Add to memory cache
                self._add_to_memory_cache(cache_key, explanation)

                # Cleanup if cache is too large
                self._cleanup_if_needed()
                self._save_index()

            except Exception as e:
                logger.warning(f"Failed to cache explanation: {e}")

    def _remove_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        if cache_file.exists():
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        if cache_key in self._index:
            del self._index[cache_key]

        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

    def _cleanup_if_needed(self) -> None:
        """Remove least recently used entries if cache is too large."""
        if len(self._index) <= self.max_size:
            return

        # Sort by access count and age (prefer recently accessed, then recently created)
        current_time = time.time()
        entries_with_scores = []
        for key, entry in self._index.items():
            # Score = access_count * 10 + (current_time - last_access) / 3600 (hours)
            access_score = entry.get("access_count", 0) * 10
            age_penalty = (current_time - entry.get("last_access", entry.get("timestamp", current_time))) / 3600
            score = access_score - age_penalty
            entries_with_scores.append((score, key))

        # Sort by score (lowest first - least valuable)
        entries_with_scores.sort()

        # Remove oldest entries
        to_remove = entries_with_scores[:len(self._index) - self.max_size + 1]
        for _, key in to_remove:
            self._remove_entry(key)

    def clear(self) -> None:
        """Clear all cached explanations."""
        try:
            for cache_file in self.cache_dir.glob("*.txt"):
                cache_file.unlink()
            self._index.clear()
            self._save_index()
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def size(self) -> int:
        """Get the number of cached explanations."""
        return len(self._index)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._index:
            return {"size": 0, "total_access_count": 0}

        total_access = sum(entry["access_count"] for entry in self._index.values())
        return {
            "size": len(self._index),
            "total_access_count": total_access,
            "avg_access_count": total_access / len(self._index),
            "strategies": list(set(entry["strategy"] for entry in self._index.values())),
            "models": list(set(entry["model_name"] for entry in self._index.values()))
        }


class EmbeddingCache:
    """Cache for code embeddings to speed up similarity search."""

    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, code: str, model_name: str) -> str:
        """Generate a cache key for the given code and model."""
        content = f"{code}|{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, code: str, model_name: str) -> Optional[Any]:
        """Get cached embedding if available.

        Args:
            code: The code to get embedding for
            model_name: The embedding model name

        Returns:
            Cached embedding if available, None otherwise
        """
        cache_key = self._get_cache_key(code, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")
            return None

    def put(self, code: str, model_name: str, embedding: Any) -> None:
        """Cache an embedding.

        Args:
            code: The code that was embedded
            model_name: The embedding model name
            embedding: The embedding to cache
        """
        cache_key = self._get_cache_key(code, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear embedding cache: {e}")
