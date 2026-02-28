"""Cache utility functions.

Optimized for performance with:
- xxhash for faster hashing (with hashlib fallback) via shared hashing module
- Pre-allocated compression buffers
- Monotonic time for flush-interval tracking; wall clock for TTL expiry
- Efficient key generation with minimal allocations
"""

import gzip
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

# Re-use the shared fast-hash implementation (avoids duplicating
# the xxhash/hashlib fallback pattern across the codebase).
from ..utils.hashing import fast_hash_bytes as _fast_hash

# Use null byte separator to prevent key collisions.
# A pipe '|' is ambiguous: ('a|b', 'c') and ('a', 'b|c') produce the
# same key. Null bytes cannot appear in normal text, making collisions
# impossible.
_KEY_SEPARATOR = "\x00"
@lru_cache(maxsize=4096)
def generate_cache_key(*components: str) -> str:
    """Generate a cache key from components (cached for repeated lookups).
    
    Uses xxhash for speed when available (10x faster than SHA256).
    Null-byte separator prevents ambiguous key collisions.
    """
    content = _KEY_SEPARATOR.join(str(c) for c in components)
    return _fast_hash(content.encode('utf-8'))

# Use monotonic time for TTL calculations (more reliable than wall clock)


def is_expired(timestamp: float, ttl_seconds: int) -> bool:
    """Check if a wall-clock timestamp is past its TTL.

    Args:
        timestamp: A value previously obtained from time.time() and stored
                   in the cache index.
        ttl_seconds: Maximum allowed age in seconds.

    Returns:
        True if the entry has expired, False otherwise.

    Note: Uses time.time() (wall clock) because cached timestamps are
    themselves recorded with time.time().  The monotonic clock is used
    only for internal flush-interval tracking (self._last_flush_time) to
    avoid sensitivity to system clock adjustments.
    """
    return time.time() - timestamp > ttl_seconds


# Compression level 6 is a good balance of speed vs compression ratio
_COMPRESSION_LEVEL = 6


def compress_data(data: str, min_size: int = 1000) -> Tuple[bytes, bool]:
    """Compress data if beneficial.
    
    Returns (data_bytes, is_compressed) tuple for efficient decompression.
    """
    data_bytes = data.encode('utf-8')
    if len(data_bytes) < min_size:
        return data_bytes, False
    
    compressed = gzip.compress(data_bytes, compresslevel=_COMPRESSION_LEVEL)
    # Only use compression if it actually saves space
    if len(compressed) < len(data_bytes) * 0.9:  # At least 10% savings
        return compressed, True
    return data_bytes, False


def decompress_data(data: bytes, is_compressed: bool = True) -> str:
    """Decompress data if compressed.
    
    Uses is_compressed flag to avoid gzip header check overhead.
    """
    if not is_compressed:
        return data.decode('utf-8')
    
    try:
        decompressed = gzip.decompress(data)
        return decompressed.decode('utf-8')
    except (gzip.BadGzipFile, OSError):
        # Fallback: assume uncompressed
        return data.decode('utf-8')


def safe_file_operation(file_path: Path, mode: str = 'r',
                       data: Optional[Any] = None) -> Optional[Any]:
    """Safely perform file operations with error handling."""
    try:
        if 'w' in mode:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, mode) as f:
                f.write(data)
            return True
        else:
            with open(file_path, mode) as f:
                return f.read()
    except (OSError, IOError, PermissionError):
        # Silent failure with None return for caller to handle
        return None


def calculate_cache_score(entry: dict, current_time: float) -> float:
    """Calculate cache entry score for LRU eviction.
    
    Higher scores = more valuable entries to keep.
    Uses weighted combination of access count and recency.
    """
    access_count = entry.get("access_count", 0)
    last_access = entry.get("last_access", entry.get("timestamp", current_time))
    
    # Weighted scoring: frequency (10x weight) - age penalty
    access_score = access_count * 10
    age_hours = (current_time - last_access) / 3600
    age_penalty = min(age_hours, 24)  # Cap penalty at 24 hours
    
    return access_score - age_penalty


def ensure_directory(path: Path) -> bool:
    """Ensure a directory exists. Returns True on success."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError):
        return False