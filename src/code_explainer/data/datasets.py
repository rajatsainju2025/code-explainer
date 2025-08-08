"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DatasetConfig:
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    test_file: Optional[str] = None
    max_examples: Optional[int] = None


def _limit(items: List[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    return items[:n] if n else items


def load_from_json(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return _limit(data, max_examples)


def load_from_csv(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    raw_records = df.to_dict(orient="records")
    # Ensure keys are strings for typing sanity
    records: List[Dict[str, Any]] = [
        {str(k): v for k, v in rec.items()} for rec in raw_records  # type: ignore[call-arg]
    ]
    return _limit(records, max_examples)


def load_from_hf(hub_id: str, split: str = "train", max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    # Lazy import to avoid optional dependency/type checking issues
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(hub_id, split=split)
    data: List[Dict[str, Any]] = ds.to_list()
    return _limit(data, max_examples)


def build_dataset_dict(config: DatasetConfig) -> Dict[str, Any]:
    """Build a simple dict with optional train/eval/test lists of dicts.
    Caller (trainer) can convert to HF Dataset if desired.
    """
    parts: Dict[str, Any] = {}

    if config.train_file and Path(config.train_file).suffix.lower() == ".json":
        parts["train"] = load_from_json(config.train_file, config.max_examples)
    if config.eval_file and Path(config.eval_file).suffix.lower() == ".json":
        parts["eval"] = load_from_json(config.eval_file, config.max_examples)
    if config.test_file and Path(config.test_file).suffix.lower() == ".json":
        parts["test"] = load_from_json(config.test_file, config.max_examples)

    if not parts:
        demo = [
            {"code": "def add(a,b): return a+b", "explanation": "Adds two numbers."},
            {"code": "x=[1,2]; x.append(3)", "explanation": "Creates a list and appends 3."},
        ]
        parts["train"] = demo
        parts["eval"] = demo

    return parts
