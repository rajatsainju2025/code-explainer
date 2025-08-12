"""Caching utilities for code explanations and embeddings."""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class ExplanationCache:
    """Cache for code explanations to avoid redundant model calls."""

    def __init__(self, cache_dir: str = ".cache/explanations", max_size: int = 1000):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached explanations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._index_file = self.cache_dir / "index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the cache index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self) -> None:
        """Save the cache index."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _get_cache_key(self, code: str, strategy: str, model_name: str) -> str:
        """Generate a cache key for the given parameters."""
        content = f"{code}|{strategy}|{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, code: str, strategy: str, model_name: str) -> Optional[str]:
        """Get a cached explanation if available.
        
        Args:
            code: The code to explain
            strategy: The prompting strategy used
            model_name: The model name used
            
        Returns:
            Cached explanation if available, None otherwise
        """
        cache_key = self._get_cache_key(code, strategy, model_name)
        
        if cache_key not in self._index:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.txt"
        if not cache_file.exists():
            # Remove stale index entry
            del self._index[cache_key]
            self._save_index()
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                explanation = f.read()
            
            # Update access time
            self._index[cache_key]["access_count"] += 1
            self._save_index()
            
            return explanation
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None
    
    def put(self, code: str, strategy: str, model_name: str, explanation: str) -> None:
        """Cache an explanation.
        
        Args:
            code: The code that was explained
            strategy: The prompting strategy used
            model_name: The model name used
            explanation: The explanation to cache
        """
        cache_key = self._get_cache_key(code, strategy, model_name)
        cache_file = self.cache_dir / f"{cache_key}.txt"
        
        try:
            # Write explanation to file
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(explanation)
            
            # Update index
            self._index[cache_key] = {
                "access_count": 1,
                "code_length": len(code),
                "strategy": strategy,
                "model_name": model_name
            }
            
            # Cleanup if cache is too large
            self._cleanup_if_needed()
            self._save_index()
            
        except Exception as e:
            logger.warning(f"Failed to cache explanation: {e}")
    
    def _cleanup_if_needed(self) -> None:
        """Remove least recently used entries if cache is too large."""
        if len(self._index) <= self.max_size:
            return
        
        # Sort by access count (ascending)
        sorted_keys = sorted(
            self._index.keys(),
            key=lambda k: self._index[k]["access_count"]
        )
        
        # Remove oldest entries
        to_remove = sorted_keys[:len(self._index) - self.max_size + 1]
        for key in to_remove:
            cache_file = self.cache_dir / f"{key}.txt"
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
            
            if key in self._index:
                del self._index[key]
    
    def clear(self) -> None:
        """Clear all cached explanations."""
        try:
            for cache_file in self.cache_dir.glob("*.txt"):
                cache_file.unlink()
            self._index.clear()
            self._save_index()
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def size(self) -> int:
        """Get the number of cached explanations."""
        return len(self._index)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._index:
            return {"size": 0, "total_access_count": 0}
        
        total_access = sum(entry["access_count"] for entry in self._index.values())
        return {
            "size": len(self._index),
            "total_access_count": total_access,
            "avg_access_count": total_access / len(self._index),
            "strategies": list(set(entry["strategy"] for entry in self._index.values())),
            "models": list(set(entry["model_name"] for entry in self._index.values()))
        }


class EmbeddingCache:
    """Cache for code embeddings to speed up similarity search."""

    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, code: str, model_name: str) -> str:
        """Generate a cache key for the given code and model."""
        content = f"{code}|{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, code: str, model_name: str) -> Optional[Any]:
        """Get cached embedding if available.
        
        Args:
            code: The code to get embedding for
            model_name: The embedding model name
            
        Returns:
            Cached embedding if available, None otherwise
        """
        cache_key = self._get_cache_key(code, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")
            return None
    
    def put(self, code: str, model_name: str, embedding: Any) -> None:
        """Cache an embedding.
        
        Args:
            code: The code that was embedded
            model_name: The embedding model name
            embedding: The embedding to cache
        """
        cache_key = self._get_cache_key(code, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear embedding cache: {e}")
