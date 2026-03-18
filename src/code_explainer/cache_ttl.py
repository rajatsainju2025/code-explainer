"""Cache TTL (Time-To-Live) configuration and management.

Defines TTL constants and cache expiration policies.
"""

import time
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Final


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


class TTLCache:
    """Simple TTL-aware cache wrapper.
    
    Stores items with expiration timestamps and automatically
    invalidates expired entries on access.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """Initialize cache with TTL in seconds.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        elapsed = time.time() - timestamp
        
        if elapsed > self.ttl_seconds:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get number of items in cache (including expired).
        
        Returns:
            Total number of cache entries
        """
        return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            k for k, (_, ts) in self._cache.items()
            if (now - ts) > self.ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)
