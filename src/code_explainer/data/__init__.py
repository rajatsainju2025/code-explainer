"""Data module for dataset loading and processing.

Provides utilities for loading, processing, and managing datasets
for code explanation training and evaluation.
"""
from __future__ import annotations

from .datasets import DatasetConfig, build_dataset_dict, load_from_json

__all__ = [
    "DatasetConfig",
    "build_dataset_dict", 
    "load_from_json",
]
