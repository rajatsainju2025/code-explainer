"""Structured logging utilities for code-explainer."""

import logging
import sys
from typing import Any, Dict, Optional


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger with consistent formatting.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def log_operation(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log an operation with optional structured metadata.
    
    Args:
        logger: Logger instance
        operation: Operation description
        level: Log level (default: INFO)
        metadata: Optional dict of structured data to append
    """
    msg = operation
    if metadata:
        msg += " | " + " ".join(f"{k}={v}" for k, v in metadata.items())
    logger.log(level, msg)


def log_error(
    logger: logging.Logger,
    operation: str,
    error: Exception,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log an error with exception context.
    
    Args:
        logger: Logger instance
        operation: Operation description
        error: Exception raised
        metadata: Optional dict of structured data
    """
    msg = f"{operation}: {type(error).__name__}: {str(error)}"
    if metadata:
        msg += " | " + " ".join(f"{k}={v}" for k, v in metadata.items())
    logger.error(msg, exc_info=True)
