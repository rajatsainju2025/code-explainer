# Cross-Device Compatibility Improvement Plan

## Current State Analysis

### Device Handling Patterns Found:
1. **utils.get_device()** - Returns string ("cuda", "mps", "cpu") but underutilized
2. **ModelLoader._setup_device()** - Duplicates logic, creates torch.device objects directly
3. **CodeExplainer** - Mixes device string/torch.device usage: `torch.device(self.device)` 
4. **Trainer** - Uses get_device() but still has hardcoded device logic

### Problems Identified:
- **Inconsistent API**: Some methods expect strings, others torch.device objects
- **Duplicated logic**: Device detection scattered across multiple modules
- **No fallback strategy**: Code fails rather than gracefully degrading (e.g., CPU fallback when GPU OOM)
- **No environment configuration**: Users can't override device selection via env vars or config
- **Missing quantization hints**: No automatic 8-bit/16-bit selection based on device capabilities
- **No device memory management**: No checks for available VRAM or graceful handling of memory issues

## Proposed Solution Architecture

### 1. Unified Device Manager (`src/code_explainer/device_manager.py`)
```python
@dataclass
class DeviceCapabilities:
    device_type: str  # "cuda", "mps", "cpu"
    device: torch.device
    supports_8bit: bool
    supports_fp16: bool
    supports_bf16: bool
    memory_gb: Optional[float]
    compute_capability: Optional[str]  # For CUDA
    
class DeviceManager:
    def get_optimal_device(self, prefer_device: Optional[str] = None) -> DeviceCapabilities
    def get_recommended_dtype(self, device_caps: DeviceCapabilities) -> torch.dtype
    def handle_oom(self, error: Exception, current_config: ModelConfig) -> ModelConfig
    def validate_device_compatibility(self, model_name: str, device: str) -> bool
```

### 2. Environment Variable Support
- `CODE_EXPLAINER_DEVICE=cpu|cuda|mps|auto` - Force specific device
- `CODE_EXPLAINER_PRECISION=fp32|fp16|bf16|8bit|auto` - Force precision
- `CODE_EXPLAINER_FALLBACK_ENABLED=true|false` - Enable CPU fallback on GPU errors

### 3. Configuration Extensions (`configs/default.yaml`)
```yaml
device:
  preferred: "auto"  # auto, cuda, mps, cpu
  precision: "auto"  # auto, fp32, fp16, bf16, 8bit
  fallback_enabled: true
  memory_fraction: 0.8  # For CUDA
  enable_optimizations: true  # torch.compile, etc.
```

### 4. Integration Points
- **ModelLoader**: Use DeviceManager instead of _setup_device()
- **CodeExplainer**: Always work with DeviceCapabilities objects
- **Trainer**: Leverage device manager for distributed/multi-GPU scenarios
- **Benchmarks**: Standardize device testing across CPU/GPU environments

## Implementation Steps
1. Create DeviceManager class with comprehensive device detection
2. Update ModelConfig to include device preferences
3. Refactor ModelLoader to use DeviceManager
4. Update CodeExplainer device property methods
5. Add environment variable parsing
6. Update default configs with device section
7. Add device compatibility validation
8. Implement graceful fallback mechanisms

## Benefits
- **Consistency**: Single source of truth for device logic
- **Flexibility**: Easy to test on different devices or force CPU usage
- **Robustness**: Graceful degradation when preferred device unavailable
- **Performance**: Automatic selection of optimal precision/quantization per device
- **Debugging**: Better error messages and device capability reporting
- **CI/Testing**: Easier to mock device scenarios for comprehensive testing