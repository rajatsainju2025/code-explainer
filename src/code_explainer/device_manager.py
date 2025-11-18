"""
Device Management and Auto-Detection Utilities

This module provides unified device detection, capability assessment, and
configuration management for PyTorch models across CPU, CUDA, and MPS devices.
Includes persistent caching of device capabilities to avoid repeated probes.
"""

import os
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceCapabilities:
    """Comprehensive device capability information."""
    device_type: str  # "cuda", "mps", "cpu"
    device: torch.device
    supports_8bit: bool = False
    supports_fp16: bool = False
    supports_bf16: bool = False
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None  # For CUDA devices
    device_name: Optional[str] = None


class DeviceManager:
    """Centralized device detection and management.
    
    Caches device capabilities to ~/.cache/code-explainer/device_cache.json
    to avoid repeated CUDA/MPS probes in subsequent runs.
    """

    CACHE_DIR = Path.home() / ".cache" / "code-explainer"
    CACHE_FILE = CACHE_DIR / "device_cache.json"

    def __init__(self):
        self._cached_capabilities: Dict[str, DeviceCapabilities] = {}
        self._load_cached_capabilities()

    def _load_cached_capabilities(self) -> None:
        """Load cached device capabilities from disk if available."""
        if not self.CACHE_FILE.exists():
            return
        
        try:
            with open(self.CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            for device_type, data in cache_data.items():
                try:
                    # Reconstruct DeviceCapabilities from cached data
                    device = torch.device(data['device_type'])
                    cap = DeviceCapabilities(
                        device_type=data['device_type'],
                        device=device,
                        supports_8bit=data.get('supports_8bit', False),
                        supports_fp16=data.get('supports_fp16', False),
                        supports_bf16=data.get('supports_bf16', False),
                        memory_gb=data.get('memory_gb'),
                        compute_capability=data.get('compute_capability'),
                        device_name=data.get('device_name')
                    )
                    self._cached_capabilities[device_type] = cap
                    logger.debug(f"Loaded cached {device_type} capabilities from disk")
                except Exception as e:
                    logger.warning(f"Failed to load cached {device_type} capabilities: {e}")
        except Exception as e:
            logger.warning(f"Failed to load device cache file: {e}")

    def _save_cached_capabilities(self) -> None:
        """Save device capabilities to disk cache."""
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            cache_data = {}
            for device_type, cap in self._cached_capabilities.items():
                # Convert to serializable format (avoid torch.device serialization)
                cache_data[device_type] = {
                    'device_type': cap.device_type,
                    'supports_8bit': cap.supports_8bit,
                    'supports_fp16': cap.supports_fp16,
                    'supports_bf16': cap.supports_bf16,
                    'memory_gb': cap.memory_gb,
                    'compute_capability': cap.compute_capability,
                    'device_name': cap.device_name
                }
            
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
            
            logger.debug(f"Saved device capabilities cache to {self.CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save device cache file: {e}")

    def get_optimal_device(self, prefer_device: Optional[str] = None) -> DeviceCapabilities:
        """Get optimal device with fallback strategy."""
        # Check environment variable override
        env_device = os.getenv('CODE_EXPLAINER_DEVICE', '').lower()
        if env_device in ['cpu', 'cuda', 'mps', 'auto']:
            prefer_device = env_device if env_device != 'auto' else prefer_device

        # Default preference order: cuda > mps > cpu
        device_order = ['cuda', 'mps', 'cpu']
        if prefer_device and prefer_device in device_order:
            # Move preferred device to front
            device_order.remove(prefer_device)
            device_order.insert(0, prefer_device)

        for device_type in device_order:
            capabilities = self._get_device_capabilities(device_type)
            if capabilities:
                logger.info(f"Selected device: {capabilities.device} ({capabilities.device_type})")
                if capabilities.memory_gb:
                    logger.info(f"Available memory: {capabilities.memory_gb:.1f} GB")
                return capabilities

        # Fallback to CPU (should always work)
        capabilities = self._get_device_capabilities('cpu')
        if capabilities:
            logger.warning("Falling back to CPU device")
            return capabilities

        raise RuntimeError("No compatible device found")

    def _get_device_capabilities(self, device_type: str) -> Optional[DeviceCapabilities]:
        """Get capabilities for a specific device type with persistent caching."""
        # Return cached result if available (O(1) lookup)
        cached = self._cached_capabilities.get(device_type)
        if cached is not None:
            return cached

        capabilities = None

        try:
            if device_type == 'cuda' and torch.cuda.is_available():
                capabilities = self._analyze_cuda_device()
            elif device_type == 'mps' and torch.backends.mps.is_available():
                capabilities = self._analyze_mps_device()
            elif device_type == 'cpu':
                capabilities = self._analyze_cpu_device()

        except Exception as e:
            logger.warning(f"Failed to analyze {device_type} device: {e}")

        if capabilities:
            self._cached_capabilities[device_type] = capabilities
            # Save to persistent cache for future runs
            self._save_cached_capabilities()

        return capabilities

    def _analyze_cuda_device(self) -> DeviceCapabilities:
        """Analyze CUDA device capabilities."""
        device = torch.device('cuda')
        device_props = torch.cuda.get_device_properties(0)

        # Memory in GB
        memory_gb = device_props.total_memory / (1024**3)

        # Compute capability
        compute_capability = f"{device_props.major}.{device_props.minor}"

        # Check precision support
        supports_fp16 = device_props.major >= 6  # Pascal and newer
        supports_bf16 = device_props.major >= 8  # Ampere and newer
        supports_8bit = supports_fp16  # Approximate - depends on specific libraries

        return DeviceCapabilities(
            device_type='cuda',
            device=device,
            supports_8bit=supports_8bit,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            memory_gb=memory_gb,
            compute_capability=compute_capability,
            device_name=device_props.name
        )

    def _analyze_mps_device(self) -> DeviceCapabilities:
        """Analyze MPS (Apple Silicon) device capabilities."""
        device = torch.device('mps')

        # MPS generally supports fp16 but not bf16 or 8bit reliably
        return DeviceCapabilities(
            device_type='mps',
            device=device,
            supports_8bit=False,  # Limited support, often causes issues
            supports_fp16=True,
            supports_bf16=False,  # Not supported on Apple Silicon
            memory_gb=None,  # Shared memory, hard to determine
            compute_capability=None,
            device_name="Apple Silicon GPU"
        )

    def _analyze_cpu_device(self) -> DeviceCapabilities:
        """Analyze CPU device capabilities."""
        device = torch.device('cpu')

        return DeviceCapabilities(
            device_type='cpu',
            device=device,
            supports_8bit=True,  # CPU supports various quantization
            supports_fp16=True,  # CPU can handle fp16 (slower than fp32)
            supports_bf16=hasattr(torch, 'bfloat16'),  # Simple check for bf16 support
            memory_gb=None,  # System RAM, varies
            compute_capability=None,
            device_name="CPU"
        )

    def get_recommended_dtype(self, device_caps: DeviceCapabilities,
                            prefer_precision: Optional[str] = None) -> torch.dtype:
        """Get recommended dtype for a device."""
        # Check environment variable override
        env_precision = os.getenv('CODE_EXPLAINER_PRECISION', '').lower()
        if env_precision in ['fp32', 'fp16', 'bf16', '8bit', 'auto']:
            prefer_precision = env_precision if env_precision != 'auto' else prefer_precision

        # Handle explicit precision requests
        if prefer_precision == 'fp32':
            return torch.float32
        elif prefer_precision == 'fp16' and device_caps.supports_fp16:
            return torch.float16
        elif prefer_precision == 'bf16' and device_caps.supports_bf16:
            return torch.bfloat16
        elif prefer_precision == '8bit':
            if not device_caps.supports_8bit:
                logger.warning("8-bit quantization not supported, falling back to fp16")
            # 8-bit is handled separately in model loading
            return torch.float16 if device_caps.supports_fp16 else torch.float32

        # Auto-select based on device capabilities
        if device_caps.device_type == 'cuda':
            # Prefer bf16 on newer cards, fp16 on older
            if device_caps.supports_bf16:
                return torch.bfloat16
            elif device_caps.supports_fp16:
                return torch.float16
            else:
                return torch.float32
        elif device_caps.device_type == 'mps':
            # MPS works best with fp16 for performance
            return torch.float16 if device_caps.supports_fp16 else torch.float32
        else:  # CPU
            # CPU usually best with fp32, but fp16 can save memory
            return torch.float32

    def should_use_quantization(self, device_caps: DeviceCapabilities) -> bool:
        """Determine if 8-bit quantization should be used."""
        env_precision = os.getenv('CODE_EXPLAINER_PRECISION', '').lower()
        if env_precision == '8bit':
            return device_caps.supports_8bit

        # Auto-decide based on device memory (if available)
        if device_caps.memory_gb and device_caps.memory_gb < 8.0:
            return device_caps.supports_8bit

        return False

    def validate_device_compatibility(self, model_name: str, device_type: str) -> bool:
        """Check if a model is compatible with a device."""
        capabilities = self._get_device_capabilities(device_type)
        if not capabilities:
            return False

        # Basic compatibility checks
        if device_type == 'mps':
            # Some models have issues with MPS
            problematic_models = ['gpt-j', 'gpt-neox']  # Add more as discovered
            if any(model in model_name.lower() for model in problematic_models):
                logger.warning(f"Model {model_name} may have issues with MPS")
                return False

        return True

    def handle_oom_error(self, error: Exception, current_device: str) -> Optional[DeviceCapabilities]:
        """Handle out-of-memory errors with fallback strategy."""
        fallback_enabled = os.getenv('CODE_EXPLAINER_FALLBACK_ENABLED', 'true').lower() == 'true'

        if not fallback_enabled:
            logger.error("OOM fallback disabled, re-raising error")
            raise error

        logger.warning(f"OOM error on {current_device}: {error}")

        # Try fallback devices
        fallback_order = {
            'cuda': ['mps', 'cpu'],
            'mps': ['cpu'],
            'cpu': []  # No fallback from CPU
        }.get(current_device, [])

        for fallback_device in fallback_order:
            capabilities = self._get_device_capabilities(fallback_device)
            if capabilities:
                logger.info(f"Falling back to {fallback_device}")
                return capabilities

        logger.error("No fallback device available")
        return None

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information for debugging."""
        info = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'devices': {}
        }

        for device_type in ['cuda', 'mps', 'cpu']:
            capabilities = self._get_device_capabilities(device_type)
            if capabilities:
                info['devices'][device_type] = {
                    'device_name': capabilities.device_name,
                    'supports_fp16': capabilities.supports_fp16,
                    'supports_bf16': capabilities.supports_bf16,
                    'supports_8bit': capabilities.supports_8bit,
                    'memory_gb': capabilities.memory_gb,
                    'compute_capability': capabilities.compute_capability
                }

        return info


# Global device manager instance
device_manager = DeviceManager()


def get_device_capabilities(prefer_device: Optional[str] = None) -> DeviceCapabilities:
    """Get optimal device capabilities (convenience function)."""
    return device_manager.get_optimal_device(prefer_device)


def get_recommended_dtype(device_caps: DeviceCapabilities,
                         prefer_precision: Optional[str] = None) -> torch.dtype:
    """Get recommended dtype for device (convenience function)."""
    return device_manager.get_recommended_dtype(device_caps, prefer_precision)