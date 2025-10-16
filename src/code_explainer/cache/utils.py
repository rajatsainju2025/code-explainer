"""Cache utility functions."""

import gzip
import hashlib
import time
from pathlib import Path
from typing import Any, Optional


def generate_cache_key(*components: str) -> str:
    """Generate a SHA256 cache key from components."""
    content = "|".join(str(component) for component in components)
    return hashlib.sha256(content.encode()).hexdigest()


def is_expired(timestamp: float, ttl_seconds: int) -> bool:
    """Check if a timestamp is expired based on TTL."""
    return time.time() - timestamp > ttl_seconds


def compress_data(data: str, min_size: int = 1000) -> bytes:
    """Compress data if beneficial."""
    if len(data) < min_size:
        return data.encode('utf-8')
    return gzip.compress(data.encode('utf-8'))


def decompress_data(data: bytes) -> str:
    """Decompress data if compressed."""
    try:
        decompressed = gzip.decompress(data)
        return decompressed.decode('utf-8')
    except gzip.BadGzipFile:
        return data.decode('utf-8')


def safe_file_operation(operation: str, file_path: Path, mode: str = 'r',
                       data: Optional[Any] = None) -> Optional[Any]:
    """Safely perform file operations with error handling."""
    try:
        if 'w' in mode:
            with open(file_path, mode) as f:
                if 'b' in mode:
                    f.write(data)
                else:
                    f.write(data)
        else:
            with open(file_path, mode) as f:
                return f.read()
    except Exception as e:
        # In a real implementation, this would use proper logging
        print(f"Failed to {operation} file {file_path}: {e}")
        return None


def calculate_cache_score(entry: dict, current_time: float) -> float:
    """Calculate cache entry score for LRU eviction."""
    access_score = entry.get("access_count", 0) * 10
    age_penalty = (current_time - entry.get("last_access", entry.get("timestamp", current_time))) / 3600
    return access_score - age_penalty


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)