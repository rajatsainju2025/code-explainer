"""Unit tests for DeviceManager fallback and quantization decisions."""

import os
from unittest.mock import patch

import torch

from code_explainer.device_manager import DeviceManager, DeviceCapabilities


def test_get_fallback_order():
    dm = DeviceManager()
    assert dm.get_fallback_order('cuda') == ['mps', 'cpu']
    assert dm.get_fallback_order('mps') == ['cpu']
    assert dm.get_fallback_order('cpu') == []


@patch.dict(os.environ, {'CODE_EXPLAINER_PRECISION': '8bit'}, clear=True)
def test_should_use_quantization_env_override():
    dm = DeviceManager()
    caps = DeviceCapabilities(device_type='cuda', device=torch.device('cuda'), supports_8bit=True)
    assert dm.should_use_quantization(caps) is True

*** End Patch