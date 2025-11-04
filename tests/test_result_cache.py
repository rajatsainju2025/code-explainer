"""Test result caching for faster test reruns and performance tracking."""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pickle


class TestResultCache:
    """Cache test results to enable quick regression detection and benchmarking."""
    
    def __init__(self, cache_dir: str = ".test_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "test_results.pkl"
        self._cache: Dict[str, Any] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cached test results from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
        except (pickle.PickleError, IOError, OSError):
            pass
    
    def get_test_hash(self, test_name: str, test_code: str) -> str:
        """Generate hash for test identification."""
        combined = f"{test_name}:{test_code}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def cache_result(
        self,
        test_name: str,
        test_code: str,
        result: Dict[str, Any],
        duration: float
    ) -> None:
        """Cache a test result."""
        test_hash = self.get_test_hash(test_name, test_code)
        self._cache[test_hash] = {
            "test_name": test_name,
            "result": result,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        self._save_cache()
    
    def get_cached_result(
        self,
        test_name: str,
        test_code: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached test result if available."""
        test_hash = self.get_test_hash(test_name, test_code)
        return self._cache.get(test_hash)
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_duration = sum(
            entry.get("duration", 0) for entry in self._cache.values()
        )
        return {
            "cached_results": len(self._cache),
            "total_cached_duration": total_duration,
            "average_duration": total_duration / len(self._cache) if self._cache else 0
        }


# Global test result cache instance
_test_result_cache = TestResultCache()


def get_test_result_cache() -> TestResultCache:
    """Get the global test result cache instance."""
    return _test_result_cache
