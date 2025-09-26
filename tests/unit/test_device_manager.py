"""Tests for device management functionality."""

import os
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import pytest
import torch

from src.code_explainer.device_manager import DeviceManager, DeviceCapabilities, device_manager


class TestDeviceManager:
    """Test device manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.device_manager = DeviceManager()
        # Clear cache between tests
        self.device_manager._cached_capabilities.clear()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_get_optimal_device_cuda_preferred(self, mock_mps, mock_cuda):
        """Test CUDA device selection when available."""
        with patch.object(self.device_manager, '_analyze_cuda_device') as mock_analyze:
            mock_capabilities = DeviceCapabilities(
                device_type='cuda',
                device=torch.device('cuda'),
                supports_fp16=True,
                supports_bf16=True,
                memory_gb=8.0
            )
            mock_analyze.return_value = mock_capabilities
            
            result = self.device_manager.get_optimal_device()
            
            assert result.device_type == 'cuda'
            assert result.device == torch.device('cuda')
            mock_analyze.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_get_optimal_device_mps_fallback(self, mock_mps, mock_cuda):
        """Test MPS device selection when CUDA unavailable."""
        with patch.object(self.device_manager, '_analyze_mps_device') as mock_analyze:
            mock_capabilities = DeviceCapabilities(
                device_type='mps',
                device=torch.device('mps'),
                supports_fp16=True,
                supports_bf16=False
            )
            mock_analyze.return_value = mock_capabilities
            
            result = self.device_manager.get_optimal_device()
            
            assert result.device_type == 'mps'
            assert result.device == torch.device('mps')
    
    @patch.dict(os.environ, {'CODE_EXPLAINER_DEVICE': 'cpu'}, clear=True)
    @patch('torch.cuda.is_available', return_value=True)
    def test_environment_variable_override(self, mock_cuda):
        """Test environment variable device override."""
        with patch.object(self.device_manager, '_analyze_cpu_device') as mock_analyze:
            mock_capabilities = DeviceCapabilities(
                device_type='cpu',
                device=torch.device('cpu'),
                supports_fp16=True
            )
            mock_analyze.return_value = mock_capabilities
            
            result = self.device_manager.get_optimal_device()
            
            assert result.device_type == 'cpu'
    
    def test_get_recommended_dtype_cuda_bf16(self):
        """Test dtype recommendation for CUDA with bf16 support."""
        capabilities = DeviceCapabilities(
            device_type='cuda',
            device=torch.device('cuda'),
            supports_bf16=True,
            supports_fp16=True
        )
        
        dtype = self.device_manager.get_recommended_dtype(capabilities)
        
        assert dtype == torch.bfloat16
    
    def test_get_recommended_dtype_mps_fp16(self):
        """Test dtype recommendation for MPS."""
        capabilities = DeviceCapabilities(
            device_type='mps',
            device=torch.device('mps'),
            supports_fp16=True,
            supports_bf16=False
        )
        
        dtype = self.device_manager.get_recommended_dtype(capabilities)
        
        assert dtype == torch.float16
    
    def test_get_recommended_dtype_cpu_fp32(self):
        """Test dtype recommendation for CPU."""
        capabilities = DeviceCapabilities(
            device_type='cpu',
            device=torch.device('cpu'),
            supports_fp16=True,
            supports_bf16=False
        )
        
        dtype = self.device_manager.get_recommended_dtype(capabilities)
        
        assert dtype == torch.float32
    
    @patch.dict(os.environ, {'CODE_EXPLAINER_PRECISION': 'fp16'}, clear=True)
    def test_precision_environment_override(self):
        """Test precision environment variable override."""
        capabilities = DeviceCapabilities(
            device_type='cuda',
            device=torch.device('cuda'),
            supports_fp16=True,
            supports_bf16=True
        )
        
        dtype = self.device_manager.get_recommended_dtype(capabilities)
        
        assert dtype == torch.float16
    
    def test_should_use_quantization_low_memory(self):
        """Test 8-bit quantization for low memory devices."""
        capabilities = DeviceCapabilities(
            device_type='cuda',
            device=torch.device('cuda'),
            supports_8bit=True,
            memory_gb=6.0  # Low memory
        )
        
        should_use = self.device_manager.should_use_quantization(capabilities)
        
        assert should_use is True
    
    def test_should_use_quantization_high_memory(self):
        """Test no quantization for high memory devices."""
        capabilities = DeviceCapabilities(
            device_type='cuda',
            device=torch.device('cuda'),
            supports_8bit=True,
            memory_gb=16.0  # High memory
        )
        
        should_use = self.device_manager.should_use_quantization(capabilities)
        
        assert should_use is False
    
    @patch.dict(os.environ, {'CODE_EXPLAINER_PRECISION': '8bit'}, clear=True)
    def test_quantization_environment_override(self):
        """Test 8-bit quantization environment variable override."""
        capabilities = DeviceCapabilities(
            device_type='cuda',
            device=torch.device('cuda'),
            supports_8bit=True,
            memory_gb=16.0
        )
        
        should_use = self.device_manager.should_use_quantization(capabilities)
        
        assert should_use is True
    
    def test_validate_device_compatibility_mps_problematic_model(self):
        """Test device compatibility validation for problematic models."""
        result = self.device_manager.validate_device_compatibility('gpt-j-6b', 'mps')
        
        assert result is False
    
    def test_validate_device_compatibility_cuda_any_model(self):
        """Test device compatibility validation for CUDA."""
        with patch.object(self.device_manager, '_get_device_capabilities') as mock_get_caps:
            mock_get_caps.return_value = DeviceCapabilities(
                device_type='cuda',
                device=torch.device('cuda')
            )
            
            result = self.device_manager.validate_device_compatibility('any-model', 'cuda')
            
            assert result is True
    
    @patch.dict(os.environ, {'CODE_EXPLAINER_FALLBACK_ENABLED': 'false'}, clear=True)
    def test_oom_error_no_fallback(self):
        """Test OOM error handling with fallback disabled."""
        error = RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError):
            self.device_manager.handle_oom_error(error, 'cuda')
    
    @patch.dict(os.environ, {'CODE_EXPLAINER_FALLBACK_ENABLED': 'true'}, clear=True)
    def test_oom_error_with_fallback(self):
        """Test OOM error handling with fallback enabled."""
        error = RuntimeError("CUDA out of memory")
        
        with patch.object(self.device_manager, '_get_device_capabilities') as mock_get_caps:
            mock_capabilities = DeviceCapabilities(
                device_type='cpu',
                device=torch.device('cpu')
            )
            mock_get_caps.return_value = mock_capabilities
            
            result = self.device_manager.handle_oom_error(error, 'cuda')
            
            assert result is not None
            assert result.device_type == 'cpu'
    
    def test_get_device_info(self):
        """Test device information gathering."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.backends.mps.is_available', return_value=False), \
             patch.object(self.device_manager, '_get_device_capabilities') as mock_get_caps:
            
            mock_capabilities = DeviceCapabilities(
                device_type='cuda',
                device=torch.device('cuda'),
                device_name='Test GPU',
                supports_fp16=True,
                supports_bf16=True,
                memory_gb=8.0
            )
            mock_get_caps.return_value = mock_capabilities
            
            info = self.device_manager.get_device_info()
            
            assert info['cuda_available'] is True
            assert info['mps_available'] is False
            assert 'devices' in info
            assert 'cuda' in info['devices']
            assert info['devices']['cuda']['device_name'] == 'Test GPU'
    
    def test_caching(self):
        """Test device capability caching."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch.object(self.device_manager, '_analyze_cuda_device') as mock_analyze:
            
            mock_capabilities = DeviceCapabilities(
                device_type='cuda',
                device=torch.device('cuda')
            )
            mock_analyze.return_value = mock_capabilities
            
            # First call should analyze
            result1 = self.device_manager._get_device_capabilities('cuda')
            # Second call should use cache
            result2 = self.device_manager._get_device_capabilities('cuda')
            
            assert result1 == result2
            mock_analyze.assert_called_once()  # Should only be called once due to caching


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_device_capabilities_function(self):
        """Test convenience function for getting device capabilities."""
        from src.code_explainer.device_manager import get_device_capabilities
        
        with patch.object(device_manager, 'get_optimal_device') as mock_get:
            mock_capabilities = DeviceCapabilities(
                device_type='cpu',
                device=torch.device('cpu')
            )
            mock_get.return_value = mock_capabilities
            
            result = get_device_capabilities()
            
            assert result.device_type == 'cpu'
            mock_get.assert_called_once_with(None)
    
    def test_get_recommended_dtype_function(self):
        """Test convenience function for getting recommended dtype."""
        from src.code_explainer.device_manager import get_recommended_dtype
        
        capabilities = DeviceCapabilities(
            device_type='cpu',
            device=torch.device('cpu')
        )
        
        with patch.object(device_manager, 'get_recommended_dtype') as mock_get:
            mock_get.return_value = torch.float32
            
            result = get_recommended_dtype(capabilities)
            
            assert result == torch.float32
            mock_get.assert_called_once_with(capabilities, None)