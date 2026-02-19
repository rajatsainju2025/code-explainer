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
    CacheError,
)


class ProcessingError(CodeExplainerError):
    """Raised for processing-related errors.

    Defined here because it's only used within the error_handling package.
    """

    pass


__all__ = [
    "CodeExplainerError",
    "ConfigurationError",
    "ModelError",
    "ProcessingError",
    "ResourceError",
    "ValidationError",
    "CacheError",
]