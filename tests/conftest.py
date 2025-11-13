# Ensure the 'code_explainer' package is importable by adding the src directory to sys.path
import os
import sys
import threading
from pathlib import Path
from typing import Generator, Dict, Any, List
from functools import lru_cache
import torch
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Add both the src directory (for `code_explainer`) and the repo root (for `src.code_explainer`)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_explainer.config import Config, ModelConfig
from code_explainer.model_loader import ModelLoader, ModelResources

# Thread-safe fixture cache for parallel test execution
_fixture_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()

@lru_cache(maxsize=1)
def _get_config_dict():
    """Get the test config dict (cached)."""
    return {
        "model": {
            "name": "microsoft/CodeGPT-small-py",
            "arch": "causal",
            "torch_dtype": "auto",
            "load_in_8bit": False,
            "max_length": 512,
        },
        "training": {
            "output_dir": "test_output",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
        },
        "cache": {
            "enabled": True,
            "directory": ".test_cache",
            "max_size": 100,
        },
        "logging": {
            "level": "DEBUG",
            "log_file": None,
        },
        "prompt": {
            "strategy": "vanilla",
            "template": "Explain this code:\n{code}\n",
        },
    }

@pytest.fixture(scope="session")
def _session_cache():
    """Session-scoped thread-safe cache for expensive resources."""
    return _fixture_cache


@pytest.fixture(scope="module")
def _module_cache():
    """Module-scoped cache for test data."""
    return {}


def _get_cached_fixture(cache_key: str, factory_func):
    """Thread-safe cached fixture creation."""
    with _cache_lock:
        if cache_key not in _fixture_cache:
            _fixture_cache[cache_key] = factory_func()
        return _fixture_cache[cache_key]


@pytest.fixture
def test_config(_session_cache):
    """Create a test configuration (module-scoped with thread-safe caching)."""
    return _get_cached_fixture("test_config", lambda: OmegaConf.create(_get_config_dict()))

@pytest.fixture
def test_model_config(test_config):
    """Create a test model configuration."""
    return test_config.model

@pytest.fixture
def temp_dir(tmp_path) -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts (with cleanup)."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(original_cwd)

@lru_cache(maxsize=1)
def _get_test_code_samples():
    """Get cached test code samples."""
    return [
        "def add(a, b):\n    return a + b",
        "print('Hello, World!')",
        """class Rectangle:
            def __init__(self, width, height):
                self.width = width
                self.height = height

            def area(self):
                return self.width * self.height""",
    ]

@pytest.fixture
def test_code_samples():
    """Return a list of test code samples (cached)."""
    return _get_test_code_samples()

@pytest.fixture(scope="function", autouse=False)
def fast_temp_dir(tmp_path):
    """Fast temporary directory without chdir (avoids race conditions in parallel tests)."""
    return tmp_path

@lru_cache(maxsize=1)
def _get_mock_tokenizer():
    """Create a cached mock tokenizer."""
    class MockTokenizer:
        def __init__(self):
            self.pad_token = "[PAD]"
            self.eos_token = "</s>"
            self.pad_token_id = 0
            self.eos_token_id = 2

        def __call__(self, text, **kwargs):
            # Simulate tokenization
            return {
                "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
            }

        def decode(self, token_ids, skip_special_tokens=True):
            # Return a dummy explanation
            return "This code adds two numbers and returns the result."

    return MockTokenizer()

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing (thread-safe cached)."""
    return _get_cached_fixture("mock_tokenizer", _get_mock_tokenizer)

@lru_cache(maxsize=1)
def _get_mock_model():
    """Create a cached mock model."""
    class MockModel:
        def __init__(self):
            self.config = type("Config", (), {"pad_token_id": 0})

        def eval(self):
            return self

        def to(self, device):
            return self

        def generate(self, **kwargs):
            import torch
            return torch.tensor([[4, 5, 6, 0, 0]])

    return MockModel()

@pytest.fixture
def mock_model():
    """Create a mock model for testing (thread-safe cached)."""
    return _get_cached_fixture("mock_model", _get_mock_model)


@pytest.fixture(scope="session", autouse=True)
def optimize_test_performance():
    """Session fixture to optimize test performance."""
    # Disable unnecessary logging during tests
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    # Pre-warm common imports
    import torch
    import numpy as np

    yield

    # Cleanup after all tests
    import gc
    gc.collect()


# Parallel test execution configuration
def pytest_configure(config):
    """Configure pytest with parallel execution settings."""
    config.addinivalue_line(
        "markers", "serial: marks tests as non-parallelizable"
    )


def pytest_collection_modifyitems(config, items: List[pytest.Item]) -> None:
    """Modify test collection to support parallel execution.
    
    Marks tests that use shared resources as serial to prevent race conditions.
    """
    serial_markers = {"serial", "integration"}
    
    for item in items:
        # Mark integration and I/O heavy tests as serial
        if any(marker in item.keywords for marker in serial_markers):
            item.add_marker(pytest.mark.serial)

