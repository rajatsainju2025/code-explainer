"""
Device Management and Auto-Detection Utilities

This module provides unified device detection, capability assessment, and
configuration management for PyTorch models across CPU, CUDA, and MPS devices.

Optimized for:
- Lazy torch import for faster initial load
- Persistent caching to avoid repeated device probes
- Environment variable configuration
- Thread-safe singleton pattern
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any, FrozenSet, TYPE_CHECKING
import logging
import threading

# Use orjson for faster JSON operations if available, fallback to stdlib
try:
    import orjson
    def json_loads(s): return orjson.loads(s)
    def json_dumps(obj): return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode()
except ImportError:
    import json
    json_loads = json.loads
    def json_dumps(obj): return json.dumps(obj, separators=(',', ':'))

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Pre-compute valid device types and precisions for O(1) lookup
_VALID_DEVICE_TYPES: FrozenSet[str] = frozenset({'cpu', 'cuda', 'mps', 'auto'})
_VALID_PRECISIONS: FrozenSet[str] = frozenset({'fp32', 'fp16', 'bf16', '8bit', 'auto'})
_DEFAULT_DEVICE_ORDER = ('cuda', 'mps', 'cpu')

# Singleton instance and lock for thread safety
_device_manager_instance: Optional["DeviceManager"] = None
_device_manager_lock = threading.Lock()

# Lazy torch import
_torch = None

def _get_torch():
    """Lazily import torch to speed up module loading."""
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


@dataclass(frozen=False)
class DeviceCapabilities:
    """Comprehensive device capability information.
    
    Note: Using regular dataclass instead of slots=True for compatibility
    with older Python versions and serialization.
    """
    device_type: str  # "cuda", "mps", "cpu"
    device: Any  # torch.device
    supports_8bit: bool = False
    supports_fp16: bool = False
    supports_bf16: bool = False
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None  # For CUDA devices
    device_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'device_type': self.device_type,
            'supports_8bit': self.supports_8bit,
            'supports_fp16': self.supports_fp16,
            'supports_bf16': self.supports_bf16,
            'memory_gb': self.memory_gb,
            'compute_capability': self.compute_capability,
            'device_name': self.device_name
        }


def get_device_manager() -> "DeviceManager":
    """Get the singleton DeviceManager instance (thread-safe)."""
    global _device_manager_instance
    if _device_manager_instance is None:
        with _device_manager_lock:
            if _device_manager_instance is None:
                _device_manager_instance = DeviceManager()
    return _device_manager_instance


class DeviceManager:
    """Centralized device detection and management.
    
    Caches device capabilities to ~/.cache/code-explainer/device_cache.json
    to avoid repeated CUDA/MPS probes in subsequent runs.
    
    Use get_device_manager() for singleton access.
    """

    CACHE_DIR = Path.home() / ".cache" / "code-explainer"
    CACHE_FILE = CACHE_DIR / "device_cache.json"

    def __init__(self):
        self._cached_capabilities: Dict[str, DeviceCapabilities] = {}
        self._lock = threading.Lock()
        self._cache_loaded = False

    def _ensure_cache_loaded(self) -> None:
        """Lazily load cached capabilities on first access."""
        if self._cache_loaded:
            return
        
        with self._lock:
            if self._cache_loaded:
                return
            self._load_cached_capabilities()
            self._cache_loaded = True

    def _load_cached_capabilities(self) -> None:
        """Load cached device capabilities from disk if available."""
        if not self.CACHE_FILE.exists():
            return
        
        torch = _get_torch()
        
        try:
            cache_data = json_loads(self.CACHE_FILE.read_bytes())
            
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
                    logger.debug("Loaded cached %s capabilities from disk", device_type)
                except Exception as e:
                    logger.warning("Failed to load cached %s capabilities: %s", device_type, e)
        except (OSError, ValueError) as e:
            logger.warning("Failed to load device cache file: %s", e)

    def _save_cached_capabilities(self) -> None:
        """Save device capabilities to disk cache."""
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                device_type: cap.to_dict()
                for device_type, cap in self._cached_capabilities.items()
            }
            
            self.CACHE_FILE.write_text(json_dumps(cache_data))
            logger.debug("Saved device capabilities cache to %s", self.CACHE_FILE)
        except (OSError, PermissionError) as e:
            logger.warning("Failed to save device cache file: %s", e)

    def get_optimal_device(self, prefer_device: Optional[str] = None) -> DeviceCapabilities:
        """Get optimal device with fallback strategy."""
        self._ensure_cache_loaded()
        
        # Check environment variable override
        env_device = os.getenv('CODE_EXPLAINER_DEVICE', '').lower()
        if env_device in _VALID_DEVICE_TYPES:
            prefer_device = env_device if env_device != 'auto' else prefer_device

        # Build device order with preference
        if prefer_device and prefer_device in _DEFAULT_DEVICE_ORDER:
            device_order = (prefer_device,) + tuple(d for d in _DEFAULT_DEVICE_ORDER if d != prefer_device)
        else:
            device_order = _DEFAULT_DEVICE_ORDER
        
        for device_type in device_order:
            capabilities = self._get_device_capabilities(device_type)
            if capabilities:
                logger.info("Selected device: %s (%s)", capabilities.device, capabilities.device_type)
                if capabilities.memory_gb:
                    logger.info("Available memory: %.1f GB", capabilities.memory_gb)
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
            logger.warning("Failed to analyze %s device: %s", device_type, e)

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
                logger.warning("Model %s may have issues with MPS", model_name)
                return False

        return True

    def handle_oom_error(self, error: Exception, current_device: str) -> Optional[DeviceCapabilities]:
        """Handle out-of-memory errors with fallback strategy."""
        fallback_enabled = os.getenv('CODE_EXPLAINER_FALLBACK_ENABLED', 'true').lower() == 'true'

        if not fallback_enabled:
            logger.error("OOM fallback disabled, re-raising error")
            raise error

        logger.warning("OOM error on %s: %s", current_device, error)

        # Try fallback devices
        fallback_order = {
            'cuda': ['mps', 'cpu'],
            'mps': ['cpu'],
            'cpu': []  # No fallback from CPU
        }.get(current_device, [])

        for fallback_device in fallback_order:
            capabilities = self._get_device_capabilities(fallback_device)
            if capabilities:
                logger.info("Falling back to %s", fallback_device)
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