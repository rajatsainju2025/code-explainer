"""Error handling module.

Provides structured logging, exception hierarchy, and error recovery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .exceptions import (
    CodeExplainerError,
    ConfigurationError,
    ModelError,
    ProcessingError,
    ResourceError,
    ValidationError,
)
from .logger import LogEntry, StructuredLogger


def setup_logging(
    log_level: str = "INFO", 
    log_file: Optional[Path | str] = None
) -> StructuredLogger:
    """Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(
        "code_explainer",
        log_level=log_level,
        log_file=log_file,
        console_output=True
    )


__all__ = [
    "CodeExplainerError",
    "ConfigurationError",
    "LogEntry",
    "ModelError",
    "ProcessingError",
    "ResourceError",
    "StructuredLogger",
    "ValidationError",
    "setup_logging",
]