"""Utilities for efficient test data loading and caching."""

import json
import gzip
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from io import StringIO


@lru_cache(maxsize=16)
def load_test_data_file(filepath: str) -> Dict[str, Any]:
    """Load and cache test data files.
    
    Supports JSON and compressed JSON files.
    Caching avoids repeated I/O during test runs.
    """
    path = Path(filepath)
    
    if not path.exists():
        return {}
    
    try:
        if filepath.endswith('.json.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        return {}
    
    return {}


@lru_cache(maxsize=32)
def get_test_fixture(fixture_name: str, fixture_data: Optional[str] = None) -> Any:
    """Get or create cached test fixtures.
    
    This reduces memory overhead by sharing fixtures across test runs.
    """
    if fixture_data:
        try:
            return json.loads(fixture_data)
        except (json.JSONDecodeError, ValueError):
            return fixture_data
    
    return None


class LazyTestDataLoader:
    """Lazy loader for test datasets to reduce memory overhead."""
    
    def __init__(self, data_source: Union[Path, str]):
        self.data_source = Path(data_source) if isinstance(data_source, str) else data_source
        self._data = None
        self._loaded = False
    
    def get(self) -> Optional[Any]:
        """Load data on first access (lazy loading)."""
        if not self._loaded:
            if self.data_source.exists():
                try:
                    with open(self.data_source, 'r', encoding='utf-8') as f:
                        self._data = json.load(f)
                except (json.JSONDecodeError, IOError, OSError):
                    self._data = None
            self._loaded = True
        
        return self._data
    
    def __bool__(self) -> bool:
        """Check if data exists without loading it."""
        return self.data_source.exists()
    
    def __repr__(self) -> str:
        return f"LazyTestDataLoader({self.data_source})"


def compress_test_data(data: Dict[str, Any], output_file: str) -> None:
    """Compress test data to reduce storage overhead."""
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(data, f)


def batch_load_test_files(directory: Path, pattern: str = "*.json") -> Dict[str, Any]:
    """Batch load multiple test files with caching."""
    results = {}
    for filepath in directory.glob(pattern):
        try:
            results[filepath.name] = load_test_data_file(str(filepath))
        except Exception:
            continue
    
    return results
