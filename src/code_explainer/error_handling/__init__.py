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
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    logger = StructuredLogger(
        "code_explainer",
        log_level=log_level,
        log_file=log_file,
        console_output=True
    )
    return logger


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
    "setup_logging",
]