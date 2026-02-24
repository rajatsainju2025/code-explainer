"""Shared hashing and JSON serialization utilities.

Consolidates duplicate xxhash/hashlib fallback patterns and orjson/json
fallback patterns that were previously copy-pasted across 5+ modules.
"""

from typing import Any

# --- Fast hashing (xxhash with hashlib fallback) ---

try:
    import xxhash

    def fast_hash_bytes(data: bytes) -> str:
        """Hash bytes using xxhash (fast) with hashlib fallback."""
        return xxhash.xxh64(data).hexdigest()

    def fast_hash_str(data: str) -> str:
        """Hash a string using xxhash (fast) with hashlib fallback."""
        return xxhash.xxh64(data.encode()).hexdigest()

except ImportError:
    import hashlib

    def fast_hash_bytes(data: bytes) -> str:
        """Hash bytes using hashlib MD5 fallback."""
        return hashlib.md5(data, usedforsecurity=False).hexdigest()

    def fast_hash_str(data: str) -> str:
        """Hash a string using hashlib MD5 fallback."""
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()


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
