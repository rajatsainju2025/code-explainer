"""
Enhanced Error Handling and Logging Module

This module provides comprehensive error handling, logging, and monitoring
capabilities for the Code Explainer system.

Key Features:
- Structured logging with multiple levels and formats
- Custom exception hierarchy for different error types
- Error recovery and retry mechanisms
- Performance monitoring and alerting
- Log aggregation and analysis
- Integration with external monitoring systems
- Configurable logging levels and outputs

Based on best practices for production Python applications.

NOTE: This module has been refactored into a modular structure.
All functionality is now available through the error_handling package.
"""

# Import everything from the new modular structure for backward compatibility
from .error_handling import (
    CodeExplainerError,
    ConfigurationError,
    ErrorHandler,
    LogEntry,
    ModelError,
    ProcessingError,
    ResourceError,
    StructuredLogger,
    ValidationError,
)

# Legacy imports for backward compatibility
import logging
import logging.handlers
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import json
import threading
import time
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Re-export for backward compatibility
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
    # Legacy functions
    "get_logger",
    "get_error_handler",
    "setup_logging",
]

# Legacy global instances and functions for backward compatibility
_logger = StructuredLogger("code_explainer")
_error_handler = ErrorHandler(_logger)

def get_logger() -> StructuredLogger:
    """Get the global logger instance (legacy function)."""
    return _logger

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance (legacy function)."""
    return _error_handler

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup global logging configuration (legacy function)."""
    global _logger
    _logger = StructuredLogger("code_explainer", log_level, log_file)
