"""Embedding cache implementation."""

import pickle
from pathlib import Path
from typing import Any, Optional

from .base_cache import BaseCache
from .models import CacheConfig, CacheStats
from .utils import generate_cache_key, safe_file_operation


class EmbeddingCache(BaseCache):
    """Cache for code embeddings to speed up similarity search."""

    def __init__(self, cache_dir: str = ".cache/embeddings"):
        config = CacheConfig(cache_dir=cache_dir)
        super().__init__(config)

    def get(self, code: str, model_name: str) -> Optional[Any]:
        """Get cached embedding if available."""
        cache_key = generate_cache_key(code, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            data = safe_file_operation("read", cache_file, "rb")
            if data:
                return pickle.loads(data)
        except Exception:
            pass
        return None

    def put(self, code: str, model_name: str, embedding: Any) -> None:
        """Cache an embedding."""
        cache_key = generate_cache_key(code, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            data = pickle.dumps(embedding)
            safe_file_operation("write", cache_file, "wb", data)
        except Exception:
            pass  # Silent failure for caching

    def clear(self) -> None:
        """Clear all cached embeddings."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)
        except Exception:
            pass

    def size(self) -> int:
        """Get the number of cached embeddings."""
        try:
            return len(list(self.cache_dir.glob("*.pkl")))
        except Exception:
            return 0

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        # Embedding cache has minimal stats for now
        return CacheStats(size=self.size())