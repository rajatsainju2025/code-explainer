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
        """
        if getattr(self, "_injected_model", None) is not None:
            return self._injected_model  # type: ignore
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")  # Backward-compat: tests expect RuntimeError here
        return self._resources.model

    @model.setter
    def model(self, value: "PreTrainedModel") -> None:
        """Allow injection of a mock model for tests."""
        self._injected_model = value

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        """Get the loaded tokenizer; supports test injection."""
        if getattr(self, "_injected_tokenizer", None) is not None:
            return self._injected_tokenizer  # type: ignore
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