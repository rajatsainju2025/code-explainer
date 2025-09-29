"""Unit tests for model loading functionality."""

import pytest
import torch
from pathlib import Path

from code_explainer.model_loader import (
    ModelLoader,
    ModelResources,
)
from code_explainer.enhanced_error_handling import ModelError, ResourceError, ConfigurationError


def test_model_loader_initialization(test_model_config):
    """Test basic ModelLoader initialization."""
    loader = ModelLoader(test_model_config)
    assert loader.config == test_model_config
    assert isinstance(loader.device, torch.device)


def test_model_resources_creation(mock_model, mock_tokenizer):
    """Test ModelResources container creation."""
    device = torch.device("cpu")
    resources = ModelResources(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=device,
        model_type="causal"
    )
    assert resources.model == mock_model
    assert resources.tokenizer == mock_tokenizer
    assert resources.device == device
    assert resources.model_type == "causal"


def test_load_nonexistent_model(test_model_config, temp_dir):
    """Test error handling when loading nonexistent model."""
    loader = ModelLoader(test_model_config)
    with pytest.raises(ConfigurationError):
        loader.load(temp_dir / "nonexistent_model")


def test_load_invalid_config(test_model_config):
    """Test error handling with invalid configuration."""
    test_model_config.arch = "invalid_arch"
    loader = ModelLoader(test_model_config)
    with pytest.raises(ConfigurationError):
        loader.load()


@pytest.mark.slow
def test_model_load_8bit(test_model_config):
    """Test 8-bit model loading configuration."""
    test_model_config.load_in_8bit = True
    loader = ModelLoader(test_model_config)

    # 8-bit loading should be disabled on CPU
    if loader.device.type == "cpu":
        resources = loader.load()
        assert not hasattr(resources.model, "is_loaded_in_8bit")
    else:
        # Skip test on GPU as we can't guarantee availability
        pytest.skip("8-bit loading tests require CPU device")