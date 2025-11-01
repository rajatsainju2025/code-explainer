"""Benchmark script for data loading performance."""

import time
import json
from pathlib import Path
from typing import List, Dict, Any

from code_explainer.data_utils import DataLoader


def benchmark_data_loading():
    """Benchmark data loading performance."""
    loader = DataLoader()

    # Create test data if it doesn't exist
    test_data = [{"code": f"def func_{i}(): pass", "explanation": f"Function {i}"} for i in range(1000)]
    test_file = Path("data/test_benchmark.json")

    if not test_file.exists():
        test_file.parent.mkdir(exist_ok=True)
        with open(test_file, 'w') as f:
            json.dump(test_data, f)

    # Benchmark loading
    start_time = time.time()
    data = loader.load_dataset("test_benchmark")
    load_time = time.time() - start_time

    # Benchmark second load (should be cached)
    start_time = time.time()
    data2 = loader.load_dataset("test_benchmark")
    cached_load_time = time.time() - start_time

    print(".4f")
    print(".4f")
    print(".2f")

    # Clean up
    test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    benchmark_data_loading()