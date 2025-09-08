# Ensure the 'code_explainer' package is importable by adding the src directory to sys.path
import sys
from pathlib import Path
import pytest
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Add both the src directory (for `code_explainer`) and the repo root (for `src.code_explainer`)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )


@pytest.fixture(scope="session")
def sample_dataset():
    """Provide sample dataset for testing."""
    return [
        {
            "code": "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "reference": "This function calculates the nth Fibonacci number using recursion.",
            "metadata": {"difficulty": "medium", "domain": "algorithms"}
        },
        {
            "code": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
            "reference": "This function calculates the factorial of n using recursion.",
            "metadata": {"difficulty": "easy", "domain": "algorithms"}
        },
        {
            "code": "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None",
            "reference": "This class defines a node for a linked list data structure.",
            "metadata": {"difficulty": "easy", "domain": "data_structures"}
        }
    ]


@pytest.fixture
def temp_output_dir():
    """Provide temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
