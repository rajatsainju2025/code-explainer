"""Model loading utilities for Code Explainer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import ModelConfig
from .enhanced_error_handling import ModelError, ConfigurationError, ResourceError
from .device_manager import DeviceManager, DeviceCapabilities

logger = logging.getLogger(__name__)


@dataclass
class ModelResources:
    """Container for loaded model resources."""
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    model_type: str
    device_capabilities: DeviceCapabilities


class ModelLoader:
    """Handles loading and initialization of models and tokenizers."""

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
        tokenizer = AutoTokenizer.from_pretrained(path)
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
        # Get precision preference from config or device manager
        precision_pref = getattr(self.config, 'precision', 'auto')
        recommended_dtype = self.device_manager.get_recommended_dtype(
            self.device_capabilities, precision_pref
        )
        
        # Check if we should use 8-bit quantization
        use_8bit = self.config.load_in_8bit or self.device_manager.should_use_quantization(
            self.device_capabilities
        )
        
        # Prepare model loading arguments
        model_kwargs: Dict[str, Any] = {"torch_dtype": recommended_dtype}
        
        if use_8bit and self.device_capabilities.supports_8bit:
            if self.device_capabilities.device_type not in ("cpu", "mps"):
                model_kwargs.update({
                    "load_in_8bit": True,
                    "device_map": self.config.device_map or "auto"
                })
                logger.info(f"Loading model with 8-bit quantization on {self.device_capabilities.device_type}")
            else:
                logger.warning(f"8-bit quantization not supported on {self.device_capabilities.device_type}, using {recommended_dtype}")
        else:
            logger.info(f"Loading model with {recommended_dtype} precision on {self.device_capabilities.device_type}")

        # Validate device compatibility
        if not self.device_manager.validate_device_compatibility(path, self.device_capabilities.device_type):
            logger.warning(f"Model {path} may have compatibility issues with {self.device_capabilities.device_type}")

        # Load appropriate model type
        if self.config.arch == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            if getattr(model.config, "pad_token_id", None) is None:
                # Use the tokenizer's pad_token_id if available
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    try:
                        # Ensure we're assigning an integer
                        pad_token_id = getattr(tokenizer, 'pad_token_id', None)
                        if isinstance(pad_token_id, int):
                            model.config.pad_token_id = pad_token_id
                    except (AttributeError, TypeError):
                        logger.warning("Could not set pad_token_id on model config")

        # Move model to device if not using 8-bit quantization
        if not use_8bit or not self.device_capabilities.supports_8bit:
            try:
                # Move model to the target device
                device_obj = self.device
                model = model.to(device_obj)  # type: ignore
            except Exception as e:
                logger.warning(f"Could not move model to device {self.device}: {e}")
                # Try OOM fallback if available
                fallback_device = self.device_manager.handle_oom_error(e, self.device_capabilities.device_type)
                if fallback_device:
                    self.device_capabilities = fallback_device
                    self.device = fallback_device.device
                    device_obj = self.device
                    model = model.to(device_obj)  # type: ignore
                else:
                    raise
            
        model.eval()
        return model