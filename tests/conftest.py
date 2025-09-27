# Ensure the 'code_explainer' package is importable by adding the src directory to sys.path
import os
import sys
from pathlib import Path
from typing import Generator
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

@pytest.fixture
def test_config() -> Config:
    """Create a test configuration."""
    config_dict = {
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
    return OmegaConf.create(config_dict)

@pytest.fixture
def test_model_config(test_config) -> ModelConfig:
    """Create a test model configuration."""
    return test_config.model

@pytest.fixture
def temp_dir(tmp_path) -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    os.chdir(tmp_path)
    yield tmp_path

@pytest.fixture
def test_code_samples() -> list[str]:
    """Return a list of test code samples."""
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
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.pad_token = "[PAD]"
            self.eos_token = "</s>"
            self.pad_token_id = 0
        
        def __call__(self, text, **kwargs):
            import torch
            # Simulate tokenization
            return {
                "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
            }
    
    return MockTokenizer()

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
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
