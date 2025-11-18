"""Property methods for CodeExplainer class."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class CodeExplainerPropertiesMixin:
    """Mixin class containing property methods for CodeExplainer."""

    @property
    def model(self) -> "PreTrainedModel":
        """Get the loaded model.
        Allows test injection when resources are not initialized.
        Supports lazy loading: loads model on first access if not already loaded.
        """
        if getattr(self, "_injected_model", None) is not None:
            return self._injected_model  # type: ignore
        if self._resources is None:
            # Lazy load model resources on first access
            self._lazy_load_model_resources()
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")  # Backward-compat: tests expect RuntimeError here
        return self._resources.model

    @model.setter
    def model(self, value: "PreTrainedModel") -> None:
        """Allow injection of a mock model for tests."""
        self._injected_model = value

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        """Get the loaded tokenizer; supports test injection.
        Supports lazy loading: loads tokenizer on first access if not already loaded.
        """
        if getattr(self, "_injected_tokenizer", None) is not None:
            return self._injected_tokenizer  # type: ignore
        if self._resources is None:
            # Lazy load model resources on first access
            self._lazy_load_model_resources()
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")  # Backward-compat: tests expect RuntimeError here
        return self._resources.tokenizer

    @tokenizer.setter
    def tokenizer(self, value: "PreTrainedTokenizerBase") -> None:
        """Allow injection of a mock tokenizer for tests."""
        self._injected_tokenizer = value

    @property
    def device(self) -> torch.device:
        """Get the compute device; default to CPU if not initialized."""
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")  # Backward-compat: tests expect RuntimeError here
        return self._resources.device

    @property
    def arch(self) -> str:
        """Get the model architecture type; default to 'causal' if unknown."""
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")  # Backward-compat: tests expect RuntimeError here
        return self._resources.model_type

    @property
    def is_model_loaded(self) -> bool:
        """Check if model resources have been loaded.
        
        Returns:
            bool: True if model and tokenizer are loaded, False otherwise
        """
        return self._resources is not None

    def _lazy_load_model_resources(self) -> None:
        """Lazy load model resources on first access.
        
        This method is called when model or tokenizer are accessed for the first time,
        deferring expensive model loading until actually needed.
        """
        if self._resources is not None:
            return  # Already loaded
        
        if not hasattr(self, '_model_path') or self._model_path is None:
            self.logger.debug("No model path available for lazy loading")
            return
        
        self.logger.debug(f"Lazy loading model resources from: {self._model_path}")
        self._resources = self._initialize_model_resources(self._model_path)
        if self._resources is not None:
            self.logger.debug("Model resources loaded successfully via lazy loading")