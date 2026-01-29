"""Model loading utilities for Code Explainer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from functools import lru_cache

from .config import ModelConfig
from .error_handling import ModelError, ConfigurationError, ResourceError
from .device_manager import DeviceManager, DeviceCapabilities

logger = logging.getLogger(__name__)


@dataclass
class ModelResources:
    """Container for loaded model resources."""
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    model_type: str
    device_capabilities: Optional[DeviceCapabilities] = None


class ModelLoader:
    """Handles loading and initialization of models and tokenizers."""
    
    __slots__ = ('config', 'device_manager', 'device_capabilities', 'device')

    def __init__(self, config: ModelConfig):
        """Initialize model loader.

        Args:
            config: Model configuration
        """
        self.config = config
        self.device_manager = DeviceManager()

        # Get device preference from config, fallback to "auto"
        device_pref = getattr(config, 'device', 'auto')
        if device_pref == 'auto':
            device_pref = None

        self.device_capabilities = self.device_manager.get_optimal_device(device_pref)
        self.device = self.device_capabilities.device

    def load(self, model_path: Optional[Union[str, Path]] = None) -> ModelResources:
        """Load model and tokenizer from path or configuration.

        Args:
            model_path: Optional path to model files. If None, uses config.name.

        Returns:
            ModelResources: Container with loaded model resources

        Raises:
            ResourceError: If model files cannot be found
            ConfigurationError: If model configuration is invalid
            ModelError: For other loading errors
        """
        path = str(model_path or self.config.name)
        try:
            tokenizer = self._load_tokenizer(path)
            model = self._load_model(path, tokenizer)

            return ModelResources(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                model_type=self.config.arch,
                device_capabilities=self.device_capabilities
            )

        except FileNotFoundError as e:
            raise ResourceError(f"Model not found at {path}: {e}") from e
        except ValueError as e:
            raise ConfigurationError(f"Invalid model configuration: {e}") from e
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}") from e

    def _load_tokenizer(self, path: str) -> PreTrainedTokenizerBase:
        """Load and configure tokenizer.

        Args:
            path: Path or name of model/tokenizer

        Returns:
            PreTrainedTokenizerBase: Configured tokenizer
        """
        tokenizer = _cached_load_tokenizer(path)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(
        self, path: str, tokenizer: PreTrainedTokenizerBase
    ) -> PreTrainedModel:
        """Load and configure model.

        Args:
            path: Path or name of model
            tokenizer: Associated tokenizer for pad token configuration

        Returns:
            PreTrainedModel: Configured model
        """
        # Validate architecture early
        if self.config.arch not in ("causal", "seq2seq"):
            raise ValueError(f"Unsupported model architecture: {self.config.arch}")

        # Get precision preference from config or device manager
        precision_pref = getattr(self.config, 'precision', 'auto')
        recommended_dtype = self.device_manager.get_recommended_dtype(
            self.device_capabilities, precision_pref
        )

        # Check if we should use 8-bit quantization
        use_8bit = self.config.load_in_8bit or self.device_manager.should_use_quantization(
            self.device_capabilities
        )

        # Prepare model loading arguments with memory optimization
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": recommended_dtype,
            "low_cpu_mem_usage": True,  # Enable memory-efficient loading
        }

        # Add 8-bit quantization only if supported
        if use_8bit and self.device_capabilities.supports_8bit:
            if self.device_capabilities.device_type not in ("cpu", "mps"):
                model_kwargs.update({
                    "load_in_8bit": True,
                    "device_map": self.config.device_map or "auto"
                })
                logger.info("Loading model with 8-bit quantization from: %s", path)
            else:
                logger.info("8-bit quantization not supported on %s, using %s precision", 
                          self.device_capabilities.device_type, recommended_dtype)
        else:
            logger.info("Loading model with %s precision on %s", 
                       recommended_dtype, self.device_capabilities.device_type)

        # Validate device compatibility
        if not self.device_manager.validate_device_compatibility(path, self.device_capabilities.device_type):
            logger.warning("Model %s may have compatibility issues with %s", path, self.device_capabilities.device_type)

        # Load model with torch.no_grad for reduced memory overhead
        with torch.no_grad():
            # Load appropriate model type
            if self.config.arch == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(path, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
                if getattr(model.config, "pad_token_id", None) is None:
                    # Use the tokenizer's pad_token_id if available
                    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
                    if isinstance(pad_token_id, int):
                        model.config.pad_token_id = pad_token_id

            # Move model to device if not using 8-bit quantization
            if not use_8bit or not self.device_capabilities.supports_8bit:
                try:
                    model = model.to(self.device)
                except Exception as e:
                    logger.warning("Could not move model to device %s: %s", self.device, e)
                    # Try OOM fallback if available
                    fallback_device = self.device_manager.handle_oom_error(e, self.device_capabilities.device_type)
                    if fallback_device:
                        self.device_capabilities = fallback_device
                        self.device = fallback_device.device
                        model = model.to(self.device)
                    else:
                        raise

        model.eval()
        return model


@lru_cache(maxsize=8)
def _cached_load_tokenizer(path: str) -> PreTrainedTokenizerBase:
    """Cached tokenizer loader to avoid repeated disk/network fetches.

    Transformers already caches files on disk; this avoids repeated
    instantiation overhead within the same process.
    """
    return AutoTokenizer.from_pretrained(path)
