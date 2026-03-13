"""Tests for quantization strategies."""

import pytest

from code_explainer.quantization import QuantizationConfig, get_quantization_config


def test_quantization_config_valid():
    config = QuantizationConfig(dtype="fp16")
    assert config.dtype == "fp16"
    assert config.enabled is False


def test_quantization_config_invalid_dtype():
    with pytest.raises(ValueError):
        QuantizationConfig(dtype="invalid")


def test_get_quantization_config_explicit_fp32():
    config = get_quantization_config(prefer_quantization="fp32")
    assert config.dtype == "fp32"
    assert config.enabled is False


def test_get_quantization_config_low_memory():
    config = get_quantization_config(memory_gb=4.0, fallback_to_fp16=True)
    assert config.dtype == "fp16"


def test_get_quantization_config_high_memory():
    config = get_quantization_config(memory_gb=16.0)
    assert config.dtype == "fp32"
