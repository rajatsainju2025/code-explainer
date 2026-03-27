# Ensure the 'code_explainer' package is importable by adding the src directory to sys.path
import os
import sys
from pathlib import Path
from typing import Generator, List
from functools import lru_cache
import torch
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Test configuration ─────────────────────────────────────────────
_CONFIG_DICT = {
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
    "cache": {"enabled": True, "directory": ".test_cache", "max_size": 100},
    "logging": {"level": "DEBUG", "log_file": None},
    "prompt": {"strategy": "vanilla", "template": "Explain this code:\n{code}\n"},
}

_TEST_CODE_SAMPLES = (
    "def add(a, b):\n    return a + b",
    "print('Hello, World!')",
    "class Rectangle:\n"
    "    def __init__(self, width, height):\n"
    "        self.width = width\n"
    "        self.height = height\n\n"
    "    def area(self):\n"
    "        return self.width * self.height",
)


@pytest.fixture(scope="session")
def test_config():
    """Session-scoped test configuration — created once for all tests."""
    return OmegaConf.create(_CONFIG_DICT)


@pytest.fixture
def test_model_config(test_config):
    """Extract model sub-config from test_config."""
    return test_config.model


@pytest.fixture
def temp_dir(tmp_path) -> Generator[Path, None, None]:
    """Temporary directory with cwd changed for the test duration."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(original_cwd)


@pytest.fixture
def test_code_samples():
    """Return test code samples."""
    return list(_TEST_CODE_SAMPLES)


@pytest.fixture(scope="function", autouse=False)
def fast_temp_dir(tmp_path):
    """Temporary directory without chdir (safe for parallel tests)."""
    return tmp_path


# ── Mock objects ───────────────────────────────────────────────────
class _MockTokenizer:
    """Lightweight mock tokenizer for unit tests."""
    __slots__ = ("pad_token", "eos_token", "pad_token_id", "eos_token_id")

    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "This code adds two numbers and returns the result."


class _MockModel:
    """Lightweight mock model for unit tests."""
    __slots__ = ("config",)

    def __init__(self):
        self.config = type("Config", (), {"pad_token_id": 0})

    def eval(self):
        return self

    def to(self, device):
        return self

    def generate(self, **kwargs):
        return torch.tensor([[4, 5, 6, 0, 0]])


@pytest.fixture
def mock_tokenizer():
    """Fresh mock tokenizer per test (no shared mutable state)."""
    return _MockTokenizer()


@pytest.fixture
def mock_model():
    """Fresh mock model per test (no shared mutable state)."""
    return _MockModel()


# ── Session-level optimizations ────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def optimize_test_performance():
    """Suppress verbose logging during test runs."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    yield
    import gc
    gc.collect()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "serial: marks tests as non-parallelizable"
    )


def pytest_collection_modifyitems(config, items: List[pytest.Item]) -> None:
    """Ensure serial-marked tests run sequentially."""
    serial_markers = {"serial", "integration"}
    for item in items:
        if any(marker in item.keywords for marker in serial_markers):
            item.add_marker(pytest.mark.serial)

