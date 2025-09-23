"""Model loading utilities for Code Explainer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import ModelConfig


@dataclass
class ModelResources:
    """Container for loaded model resources."""
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    model_type: str


class ModelLoadError(Exception):
    """Base exception for model loading errors."""
    pass


class ModelNotFoundError(ModelLoadError):
    """Raised when model file or directory is not found."""
    pass


class ModelConfigError(ModelLoadError):
    """Raised when model configuration is invalid."""
    pass


class ModelLoader:
    """Handles loading and initialization of models and tokenizers."""

    def __init__(self, config: ModelConfig):
        """Initialize model loader.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self._setup_device()

    def _setup_device(self) -> None:
        """Set up the compute device for model execution."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def load(self, model_path: Optional[Union[str, Path]] = None) -> ModelResources:
        """Load model and tokenizer from path or configuration.
        
        Args:
            model_path: Optional path to model files. If None, uses config.name.
        
        Returns:
            ModelResources: Container with loaded model resources
            
        Raises:
            ModelNotFoundError: If model files cannot be found
            ModelConfigError: If model configuration is invalid
            ModelLoadError: For other loading errors
        """
        try:
            path = str(model_path or self.config.name)
            tokenizer = self._load_tokenizer(path)
            model = self._load_model(path, tokenizer)
            
            return ModelResources(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                model_type=self.config.arch
            )
            
        except FileNotFoundError as e:
            raise ModelNotFoundError(f"Model not found at {path}: {e}") from e
        except ValueError as e:
            raise ModelConfigError(f"Invalid model configuration: {e}") from e
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

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
        # Prepare model loading arguments
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(str(self.config.torch_dtype).lower(), "auto")
        
        model_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
        if self.config.load_in_8bit:
            if self.device.type in ("cpu", "mps"):
                # 8-bit loading not supported on CPU/MPS
                pass
            else:
                model_kwargs.update({
                    "load_in_8bit": True,
                    "device_map": self.config.device_map or "auto"
                })

        # Load appropriate model type
        if self.config.arch == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id

        # Move model to device if not using 8-bit quantization
        if not self.config.load_in_8bit:
            model.to(self.device)
            
        model.eval()
        return model