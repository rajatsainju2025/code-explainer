"""Shared hashing and JSON serialization utilities.

Consolidates duplicate xxhash/hashlib fallback patterns and orjson/json
fallback patterns that were previously copy-pasted across 5+ modules.
"""

from functools import lru_cache
from typing import Any


_MAX_CACHED_HASH_INPUT_SIZE = 2048

# --- Fast hashing (xxhash with hashlib fallback) ---

try:
    import xxhash

    @lru_cache(maxsize=8192)
    def _fast_hash_bytes_cached(data: bytes) -> str:
        """Hash bytes using xxhash and cache small repeated inputs.

        Cached for repeated short inputs to avoid re-hashing identical strings.
        """
        return xxhash.xxh64(data).hexdigest()

    @lru_cache(maxsize=8192)
    def _fast_hash_str_cached(data: str) -> str:
        """Hash a string using xxhash and cache small repeated inputs.

        Cached to avoid recomputing hashes for repeated identifiers.
        """
        return xxhash.xxh64(data.encode()).hexdigest()

    def fast_hash_bytes(data: bytes) -> str:
        """Hash bytes using xxhash, caching only bounded-size inputs."""
        if len(data) <= _MAX_CACHED_HASH_INPUT_SIZE:
            return _fast_hash_bytes_cached(data)
        return xxhash.xxh64(data).hexdigest()

    def fast_hash_str(data: str) -> str:
        """Hash a string using xxhash, caching only bounded-size inputs."""
        if len(data) <= _MAX_CACHED_HASH_INPUT_SIZE:
            return _fast_hash_str_cached(data)
        return xxhash.xxh64(data.encode()).hexdigest()

except ImportError:
    import hashlib

    @lru_cache(maxsize=8192)
    def _fast_hash_bytes_cached(data: bytes) -> str:
        """Hash bytes using hashlib MD5 fallback and cache small inputs."""
        return hashlib.md5(data, usedforsecurity=False).hexdigest()

    @lru_cache(maxsize=8192)
    def _fast_hash_str_cached(data: str) -> str:
        """Hash a string using hashlib MD5 fallback and cache small inputs."""
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()

    def fast_hash_bytes(data: bytes) -> str:
        """Hash bytes using hashlib, caching only bounded-size inputs."""
        if len(data) <= _MAX_CACHED_HASH_INPUT_SIZE:
            return _fast_hash_bytes_cached(data)
        return hashlib.md5(data, usedforsecurity=False).hexdigest()

    def fast_hash_str(data: str) -> str:
        """Hash a string using hashlib, caching only bounded-size inputs."""
        if len(data) <= _MAX_CACHED_HASH_INPUT_SIZE:
            return _fast_hash_str_cached(data)
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()


fast_hash_bytes.cache_info = _fast_hash_bytes_cached.cache_info  # type: ignore[attr-defined]
fast_hash_bytes.cache_clear = _fast_hash_bytes_cached.cache_clear  # type: ignore[attr-defined]
fast_hash_str.cache_info = _fast_hash_str_cached.cache_info  # type: ignore[attr-defined]
fast_hash_str.cache_clear = _fast_hash_str_cached.cache_clear  # type: ignore[attr-defined]


# --- Fast JSON (orjson with stdlib fallback) ---

try:
    import orjson

    def json_loads(s: str | bytes) -> Any:
        """Load JSON using orjson (3-10x faster than stdlib)."""
        return orjson.loads(s)

    def json_dumps(obj: Any) -> str:
        """Dump JSON using orjson with compact (no-indent) formatting."""
        return orjson.dumps(obj).decode()

except ImportError:
    import json

    def json_loads(s: str | bytes) -> Any:  # type: ignore[misc]
        """Load JSON using stdlib."""
        return json.loads(s)

    def json_dumps(obj: Any) -> str:  # type: ignore[misc]
        """Dump JSON using stdlib with compact separators."""
        return json.dumps(obj, separators=(',', ':'))
