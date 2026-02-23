"""Exception hierarchy for Code Explainer.

Re-exports from the canonical exceptions module to avoid duplicate class
hierarchies. All exception classes are defined in code_explainer.exceptions.
"""

from ..exceptions import (
    CodeExplainerError,
    ConfigurationError,
    ModelError,
    ResourceError,
    ValidationError,
)


__all__ = [
    "CodeExplainerError",
    "ConfigurationError",
    "ModelError",
    "ResourceError",
    "ValidationError",
]