"""
Dataset utilities for loading and processing code explanation datasets.

Optimized with:
- orjson for faster JSON parsing (3-10x faster than stdlib json)
- functools.lru_cache for memoized file loading
- Pre-computed Path objects for O(1) lookups
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Use orjson if available, fallback to stdlib json
try:
    import orjson
    
    def _load_json(path: Path) -> Any:
        """Load JSON with orjson for better performance."""
        return orjson.loads(path.read_bytes())
except ImportError:
    import json
    
    def _load_json(path: Path) -> Any:
        """Load JSON with stdlib json."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Pre-computed frozenset for valid split names
_VALID_SPLITS = frozenset({"train", "eval", "test"})


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    """Configuration for dataset loading (immutable for hashability)."""
    train_file: str = "data/train.json"
    eval_file: str = "data/eval.json"
    test_file: str = "data/test.json"
    max_examples: int = -1
    
    def get_files(self) -> Tuple[Tuple[str, str], ...]:
        """Get (split_name, file_path) tuples for iteration."""
        return (
            ("train", self.train_file),
            ("eval", self.eval_file),
            ("test", self.test_file),
        )


@lru_cache(maxsize=32)
def _cached_load_json(path_str: str) -> Tuple[Any, ...]:
    """Cached JSON loading (returns tuple for immutability)."""
    path = Path(path_str)
    if not path.exists():
        return ()
    
    data = _load_json(path)
    
    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path_str}, got {type(data).__name__}")
    
    return tuple(data)  # Return tuple for cache immutability


def build_dataset_dict(config: DatasetConfig) -> Dict[str, List[Dict[str, Any]]]:
    """Build a dictionary of datasets from configuration."""
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    max_examples = config.max_examples
    
    # Single-pass iteration over all splits
    for split_name, file_path in config.get_files():
        cached_data = _cached_load_json(file_path)
        
        if max_examples > 0 and len(cached_data) > max_examples:
            datasets[split_name] = list(cached_data[:max_examples])
        else:
            datasets[split_name] = list(cached_data)
    
    return datasets


def load_from_json(path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file (with caching)."""
    return list(_cached_load_json(path))