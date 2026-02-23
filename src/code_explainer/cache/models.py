"""Data models for caching system.

Optimized with:
- frozen=True for immutable config (hashable for caching)
- slots=True for memory efficiency where applicable
- Type hints throughout
"""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class CacheStats:
    """Cache statistics."""
    size: int = 0
    total_access_count: int = 0
    avg_access_count: float = 0.0
    strategies: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to attributes."""
        return getattr(self, key)


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for cache behavior (immutable for hashability)."""
    cache_dir: str = ".cache"
    max_size: int = 1000
    ttl_seconds: int = 86400  # 24 hours
    compression_enabled: bool = True
    memory_cache_size: int = 100
    cleanup_threshold: float = 0.9  # Cleanup when 90% full