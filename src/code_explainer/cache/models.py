"""Data models for caching system."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass(slots=True)
class CacheEntry:
    """Base cache entry structure."""
    key: str
    data: Any
    timestamp: float
    access_count: int = 0
    last_access: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationEntry(CacheEntry):
    """Cache entry for code explanations."""
    __slots__ = ('code', 'strategy', 'model_name', 'compressed', 'code_length')
    code: str = ""
    strategy: str = ""
    model_name: str = ""
    compressed: bool = False
    code_length: int = 0


@dataclass
class EmbeddingEntry(CacheEntry):
    """Cache entry for code embeddings."""
    __slots__ = ('code', 'model_name')
    code: str = ""
    model_name: str = ""


@dataclass(slots=True)
class CacheStats:
    """Cache statistics."""
    size: int = 0
    total_access_count: int = 0
    avg_access_count: float = 0.0
    strategies: list = field(default_factory=list)
    models: list = field(default_factory=list)
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    def __getitem__(self, key):
        """Allow dict-like access to attributes."""
        return getattr(self, key)


@dataclass(slots=True)
class CacheConfig:
    """Configuration for cache behavior."""
    cache_dir: str = ".cache"
    max_size: int = 1000
    ttl_seconds: int = 86400  # 24 hours
    compression_enabled: bool = True
    memory_cache_size: int = 100
    cleanup_threshold: float = 0.9  # Cleanup when 90% full