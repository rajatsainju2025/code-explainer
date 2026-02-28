"""Embedding cache implementation.

Uses numpy .npy format for safe, efficient serialization of embeddings
instead of pickle (which allows arbitrary code execution on load).
"""

import logging
from typing import Any, Optional

import numpy as np

from .base_cache import BaseCache
from .models import CacheConfig, CacheStats
from .utils import generate_cache_key

logger = logging.getLogger(__name__)

# File extension for safe numpy format
_CACHE_EXT = ".npy"


class EmbeddingCache(BaseCache):
    """Cache for code embeddings to speed up similarity search.
    
    Uses numpy's .npy format instead of pickle to prevent
    arbitrary code execution from untrusted cache files.
    """

    def __init__(self, cache_dir: str = ".cache/embeddings"):
        config = CacheConfig(cache_dir=cache_dir)
        super().__init__(config)

    def get(self, code: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        cache_key = generate_cache_key(code, model_name)
        cache_file = self.cache_dir / f"{cache_key}{_CACHE_EXT}"

        if not cache_file.exists():
            return None

        try:
            # allow_pickle=False prevents code execution attacks
            return np.load(cache_file, allow_pickle=False)
        except Exception as e:
            logger.debug("Failed to load cached embedding %s: %s", cache_key, e)
            return None

    def put(self, code: str, model_name: str, embedding: Any) -> None:
        """Cache an embedding using safe numpy format."""
        cache_key = generate_cache_key(code, model_name)
        cache_file = self.cache_dir / f"{cache_key}{_CACHE_EXT}"

        try:
            arr = np.asarray(embedding)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, arr, allow_pickle=False)
        except Exception as e:
            logger.debug("Failed to cache embedding %s: %s", cache_key, e)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        try:
            for cache_file in self.cache_dir.glob(f"*{_CACHE_EXT}"):
                cache_file.unlink(missing_ok=True)
        except Exception:
            pass

    def size(self) -> int:
        """Get the number of cached embeddings."""
        try:
            return len(list(self.cache_dir.glob(f"*{_CACHE_EXT}")))
        except Exception:
            return 0

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(size=self.size())