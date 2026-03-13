"""Quantization strategy and configuration helpers.

Provides simple heuristics for when to use 8-bit quantization based on
device memory and user preferences. Serves as an adapter for bitsandbytes.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    enabled: bool = False
    dtype: str = "fp32"  # fp32, fp16, bf16, 8bit
    use_bnb_4bit: bool = False  # bitsandbytes 4-bit (optional)
    use_bnb_8bit: bool = False  # bitsandbytes 8-bit (optional)

    def __post_init__(self):
        if self.dtype not in {"fp32", "fp16", "bf16", "8bit"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")


def get_quantization_config(
    memory_gb: Optional[float] = None,
    prefer_quantization: Optional[str] = None,
    fallback_to_fp16: bool = True
) -> QuantizationConfig:
    """Decide on quantization strategy based on device memory and preference.

    Args:
        memory_gb: Available GPU memory in gigabytes (None = unknown/CPU)
        prefer_quantization: User preference ("fp32", "fp16", "bf16", "8bit")
        fallback_to_fp16: If 8-bit not available, fall back to fp16

    Returns:
        QuantizationConfig with decided settings
    """
    # Respect explicit user preference
    if prefer_quantization:
        if prefer_quantization == "8bit":
            return QuantizationConfig(enabled=True, dtype="8bit", use_bnb_8bit=True)
        else:
            return QuantizationConfig(dtype=prefer_quantization)

    # Auto-decide based on memory
    if memory_gb and memory_gb < 8.0:
        # Low memory device: try 8-bit, fall back to fp16
        config = QuantizationConfig(enabled=True, dtype="8bit", use_bnb_8bit=True)
        if fallback_to_fp16:
            config.dtype = "fp16"
        return config

    # Default: fp32 for high-memory devices
    return QuantizationConfig()
