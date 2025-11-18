"""Cross-process model cache for SentenceTransformer models."""

import json
import logging
import os
import pickle
import hashlib
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from fcntl import flock, LOCK_SH, LOCK_EX, LOCK_UN

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PersistentModelCache:
    """
    Disk-based model cache for SentenceTransformer with cross-process support.
    
    Uses file-locking to ensure thread-safe and process-safe access to cached models.
    Models are cached in ~/.cache/code-explainer/models/ directory.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize persistent model cache.
        
        Args:
            cache_dir: Optional custom cache directory. Defaults to ~/.cache/code-explainer/models/
        """
        if cache_dir is None:
            home = Path.home()
            self.cache_dir = home / ".cache" / "code-explainer" / "models"
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock_dir = self.cache_dir / ".locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._local_cache: Dict[str, SentenceTransformer] = {}
        self._local_lock = threading.Lock()
        
        logger.debug(f"Persistent model cache initialized at: {self.cache_dir}")

    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache file path for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the cache file
        """
        # Create a safe filename from model name
        safe_name = hashlib.md5(model_name.encode()).hexdigest()
        return self.cache_dir / f"{safe_name}.pkl"

    def _get_lock_path(self, model_name: str) -> Path:
        """Get lock file path for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the lock file
        """
        safe_name = hashlib.md5(model_name.encode()).hexdigest()
        return self._lock_dir / f"{safe_name}.lock"

    def get(self, model_name: str) -> Optional[SentenceTransformer]:
        """Get a model from cache if available.
        
        Args:
            model_name: Name of the model
            
        Returns:
            SentenceTransformer instance if cached, None otherwise
        """
        # Check local in-memory cache first
        with self._local_lock:
            if model_name in self._local_cache:
                logger.debug(f"Model cache hit (in-memory): {model_name}")
                return self._local_cache[model_name]

        # Check disk cache
        cache_path = self._get_cache_path(model_name)
        if not cache_path.exists():
            return None

        try:
            lock_path = self._get_lock_path(model_name)
            with open(lock_path, 'a') as lock_file:
                # Acquire shared lock (read lock)
                flock(lock_file.fileno(), LOCK_SH)
                try:
                    with open(cache_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Cache in local memory
                    with self._local_lock:
                        self._local_cache[model_name] = model
                    
                    logger.debug(f"Model cache hit (disk): {model_name}")
                    return model
                finally:
                    flock(lock_file.fileno(), LOCK_UN)
        except Exception as e:
            logger.warning(f"Failed to load model from cache: {e}")
            return None

    def put(self, model_name: str, model: SentenceTransformer) -> bool:
        """Cache a model to disk.
        
        Args:
            model_name: Name of the model
            model: SentenceTransformer instance to cache
            
        Returns:
            True if caching succeeded, False otherwise
        """
        try:
            # Store in local memory cache
            with self._local_lock:
                self._local_cache[model_name] = model

            # Store on disk
            cache_path = self._get_cache_path(model_name)
            lock_path = self._get_lock_path(model_name)
            
            with open(lock_path, 'a') as lock_file:
                # Acquire exclusive lock (write lock)
                flock(lock_file.fileno(), LOCK_EX)
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.debug(f"Model cached to disk: {model_name}")
                    return True
                finally:
                    flock(lock_file.fileno(), LOCK_UN)
        except Exception as e:
            logger.warning(f"Failed to cache model to disk: {e}")
            # Still return True as in-memory cache is available
            return False

    def clear(self) -> None:
        """Clear all caches (local and disk)."""
        with self._local_lock:
            self._local_cache.clear()
        
        # Remove disk cache files (but keep lock files for safety)
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.debug("Model cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")

    def get_cache_size(self) -> int:
        """Get approximate size of disk cache in bytes.
        
        Returns:
            Total size of cached models in bytes
        """
        total_size = 0
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                total_size += cache_file.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate cache size: {e}")
        
        return total_size

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache usage.
        
        Returns:
            Dict with cache stats (in_memory_models, disk_models, disk_size_mb)
        """
        try:
            disk_size_mb = self.get_cache_size() / (1024 * 1024)
            disk_models = list(self.cache_dir.glob("*.pkl"))
            
            return {
                "in_memory_models": len(self._local_cache),
                "disk_models": len(disk_models),
                "disk_size_mb": round(disk_size_mb, 2),
                "cache_dir": str(self.cache_dir)
            }
        except Exception as e:
            logger.warning(f"Failed to get cache info: {e}")
            return {
                "in_memory_models": len(self._local_cache),
                "disk_models": 0,
                "disk_size_mb": 0.0,
                "cache_dir": str(self.cache_dir),
                "error": str(e)
            }


# Global persistent cache instance
_PERSISTENT_CACHE = PersistentModelCache()


def get_cached_model(model_name: str) -> SentenceTransformer:
    """Get or load a model with persistent cross-process caching.
    
    Checks in-memory cache first, then disk cache, then loads fresh.
    
    Args:
        model_name: Name of the SentenceTransformer model
        
    Returns:
        SentenceTransformer instance
    """
    # Try to get from persistent cache
    cached_model = _PERSISTENT_CACHE.get(model_name)
    if cached_model is not None:
        return cached_model
    
    # Load fresh and cache
    logger.debug(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Cache for future use
    _PERSISTENT_CACHE.put(model_name, model)
    
    return model


def clear_model_cache() -> None:
    """Clear all persistent model caches."""
    _PERSISTENT_CACHE.clear()


def get_model_cache_info() -> Dict[str, Any]:
    """Get information about model cache usage.
    
    Returns:
        Dict with cache statistics
    """
    return _PERSISTENT_CACHE.get_cache_info()
