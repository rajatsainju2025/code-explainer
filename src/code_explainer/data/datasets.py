"""
Dataset utilities for loading and processing code explanation datasets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import json


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    train_file: str = "data/train.json"
    eval_file: str = "data/eval.json"
    test_file: str = "data/test.json"
    max_examples: int = -1


def build_dataset_dict(config: DatasetConfig) -> Dict[str, List[Dict[str, Any]]]:
    """Build a dictionary of datasets from configuration."""
    datasets = {}

    # Load train data
    if Path(config.train_file).exists():
        datasets["train"] = load_from_json(config.train_file)
    else:
        datasets["train"] = []

    # Load eval data
    if Path(config.eval_file).exists():
        datasets["eval"] = load_from_json(config.eval_file)
    else:
        datasets["eval"] = []

    # Load test data
    if Path(config.test_file).exists():
        datasets["test"] = load_from_json(config.test_file)
    else:
        datasets["test"] = []

    # Apply max_samples limit if specified
    if config.max_examples > 0:
        for key in datasets:
            if len(datasets[key]) > config.max_examples:
                datasets[key] = datasets[key][:config.max_examples]

    return datasets


def load_from_json(path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")

    return data