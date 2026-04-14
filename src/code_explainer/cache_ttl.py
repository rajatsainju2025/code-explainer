"""Cache TTL (Time-To-Live) configuration and management.

Defines TTL constants and cache expiration policies.
"""

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Final


__all__ = [
    "ONE_HOUR",
    "TWO_HOURS", 
    "ONE_DAY",
    "CacheTTLConfig",
    "TTLCache",
]


# Named constants for TTL durations (in seconds)
ONE_HOUR: Final[int] = 3600
TWO_HOURS: Final[int] = 7200
ONE_DAY: Final[int] = 86400


@dataclass
class CacheTTLConfig:
    """Configuration for cache expiration times.
    
    Attributes:
        embedding_cache_ttl_seconds: TTL for embedding lookup cache (default: 1 hour)
        explanation_cache_ttl_seconds: TTL for explanation result cache (default: 2 hours)
        model_info_cache_ttl_seconds: TTL for model metadata cache (default: 24 hours)
        enable_ttl_enforcement: Whether to enforce TTL expiration (default: True)
    """
    embedding_cache_ttl_seconds: int = ONE_HOUR
    explanation_cache_ttl_seconds: int = TWO_HOURS
    model_info_cache_ttl_seconds: int = ONE_DAY
    enable_ttl_enforcement: bool = True
    
    @classmethod
    def from_env(cls) -> "CacheTTLConfig":
        """Create config from environment variables.
        
        Environment variables:
            CODE_EXPLAINER_CACHE_EMBEDDING_TTL: Embedding cache TTL in seconds
            CODE_EXPLAINER_CACHE_EXPLANATION_TTL: Explanation cache TTL in seconds
            CODE_EXPLAINER_CACHE_MODEL_TTL: Model info cache TTL in seconds
            CODE_EXPLAINER_CACHE_DISABLE_TTL: Set to '1' to disable TTL enforcement
        
        Returns:
            CacheTTLConfig instance with values from environment or defaults
        """
        return cls(
            embedding_cache_ttl_seconds=int(
                os.getenv("CODE_EXPLAINER_CACHE_EMBEDDING_TTL", "3600")
            ),
            explanation_cache_ttl_seconds=int(
                os.getenv("CODE_EXPLAINER_CACHE_EXPLANATION_TTL", "7200")
            ),
            model_info_cache_ttl_seconds=int(
                os.getenv("CODE_EXPLAINER_CACHE_MODEL_TTL", "86400")
            ),
            enable_ttl_enforcement=os.getenv(
                "CODE_EXPLAINER_CACHE_DISABLE_TTL", "0"
            ) != "1",
        )

    def __repr__(self) -> str:
        return (
            f"CacheTTLConfig("
            f"embedding_ttl={self.embedding_cache_ttl_seconds}s, "
            f"explanation_ttl={self.explanation_cache_ttl_seconds}s, "
            f"model_ttl={self.model_info_cache_ttl_seconds}s, "
            f"enforce={self.enable_ttl_enforcement})"
        )


class TTLCache:
    """Simple TTL-aware cache with bounded size.
    
    Stores items with expiration timestamps. Automatically invalidates
    expired entries on access and evicts oldest entries when max_size
    is exceeded.
    """

    __slots__ = ("ttl_seconds", "max_size", "_cache")
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10_000):
        """Initialize cache with TTL and maximum size.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_size: Maximum number of entries before eviction (0 = unlimited)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value if found and not expired, None otherwise
        """
        entry = self._cache.get(key)
        if entry is None:
            return None

        value, expires_at = entry
        if time.monotonic() >= expires_at:
            self._cache.pop(key, None)
            return None

        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp.
        
        Evicts expired entries and oldest entries if over max_size.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        now = time.monotonic()
        if key in self._cache:
            self._cache.pop(key, None)
        elif self.max_size > 0 and len(self._cache) >= self.max_size:
            self.cleanup_expired(now)
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

        self._cache[key] = (value, now + self.ttl_seconds)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get number of items in cache (including expired).
        
        Returns:
            Total number of cache entries
        """
        return len(self._cache)
    
    def cleanup_expired(self, now: Optional[float] = None) -> int:
        """Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.monotonic() if now is None else now
        removed = 0

        while self._cache:
            _, (_, expires_at) = next(iter(self._cache.items()))
            if expires_at > current_time:
                break
            self._cache.popitem(last=False)
            removed += 1

        return removed

    def __repr__(self) -> str:
        cap = str(self.max_size) if self.max_size > 0 else "∞"
        return f"TTLCache(ttl={self.ttl_seconds}s, size={len(self._cache)}/{cap})"
