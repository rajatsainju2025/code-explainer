"""Error handling module."""

from .error_handler import ErrorHandler
from .exceptions import (
    CodeExplainerError,
    ConfigurationError,
    ModelError,
    ProcessingError,
    ResourceError,
    ValidationError,
)
from .logger import LogEntry, StructuredLogger

__all__ = [
    "CodeExplainerError",
    "ConfigurationError",
    "ErrorHandler",
    "LogEntry",
    "ModelError",
    "ProcessingError",
    "ResourceError",
    "StructuredLogger",
    "ValidationError",
]