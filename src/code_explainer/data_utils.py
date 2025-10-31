"""
Data utilities for dataset loading and processing.
"""

from typing import Dict, Any, List, Optional, Iterable
from pathlib import Path
import json


class DataLoader:
    """Loads and processes datasets for code explanation."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load a dataset by name."""
        dataset_path = self.data_dir / f"{dataset_name}.json"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        return data

    def iter_dataset(self, dataset_name: str) -> Iterable[Dict[str, Any]]:
        """Stream a dataset item-by-item to reduce peak memory usage.

        Supports two formats:
        - JSON array (default): falls back to json.load then yields
        - JSON Lines (.jsonl): streams line-by-line without loading entire file
        """
        dataset_json = self.data_dir / f"{dataset_name}.json"
        dataset_jsonl = self.data_dir / f"{dataset_name}.jsonl"

        # Prefer JSONL if available for true streaming
        path = dataset_jsonl if dataset_jsonl.exists() else dataset_json

        if path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        # Skip malformed lines to keep streaming robust
                        continue
        else:
            # Attempt streaming parse with ijson if available
            try:
                import ijson  # type: ignore
                with open(path, "rb") as f:
                    for item in ijson.items(f, "item"):
                        yield item
            except Exception:
                # Fallback: load once, then yield items
                data = self.load_dataset(dataset_name)
                for item in data:
                    yield item

    def load_train_data(self) -> List[Dict[str, Any]]:
        """Load training data."""
        return self.load_dataset("train")

    def load_eval_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data."""
        return self.load_dataset("eval")

    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data."""
        return self.load_dataset("test")

    def save_dataset(self, dataset_name: str, data: List[Dict[str, Any]]):
        """Save a dataset."""
        dataset_path = self.data_dir / f"{dataset_name}.json"
        self.data_dir.mkdir(exist_ok=True)

        with open(dataset_path, 'w') as f:
            json.dump(data, f, indent=2)