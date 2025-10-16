"""
Data utilities for dataset loading and processing.
"""

from typing import Dict, Any, List, Optional
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