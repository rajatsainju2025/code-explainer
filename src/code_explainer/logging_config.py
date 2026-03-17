"""Structured logging configuration for production."""

import logging
import json
import sys
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from typing import Any, Dict
import os


class StructuredLogger:
    """Structured JSON logger for production deployments."""

    def __init__(self, name: str, level: str = "INFO"):
        """Initialize structured logger.

        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level, logging.INFO))
        self.logger.handlers = []  # Clear existing handlers

        # JSON formatter for structured logging
        log_format = "%(timestamp)s %(level)s %(name)s %(message)s"
        json_formatter = jsonlogger.JsonFormatter(log_format)

        # Console handler (STDOUT)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)

        # File handler with rotation
        log_dir = os.environ.get("LOG_DIR", "/var/log/code-explainer")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "app.log")

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=10
        )
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)

        # Error file handler
        error_log_file = os.path.join(log_dir, "error.log")
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=100 * 1024 * 1024,
            backupCount=10,
            level=logging.ERROR
        )
        error_handler.setFormatter(json_formatter)
        self.logger.addHandler(error_handler)

    def _add_context(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add context to log message.

        Args:
            extra: Additional context fields

        Returns:
            Extended context dictionary
        """
        context = {
            "service": "code-explainer",
            "version": os.environ.get("APP_VERSION", "unknown"),
        }
        if extra:
            context.update(extra)
        return context

    def info(self, message: str, **extra):
        """Log info message with context."""
        self.logger.info(message, extra=self._add_context(extra))

    def debug(self, message: str, **extra):
        """Log debug message with context."""
        self.logger.debug(message, extra=self._add_context(extra))

    def warning(self, message: str, **extra):
        """Log warning message with context."""
        self.logger.warning(message, extra=self._add_context(extra))

    def error(self, message: str, exc_info=False, **extra):
        """Log error message with context and exception info."""
        self.logger.error(
            message,
            exc_info=exc_info,
            extra=self._add_context(extra)
        )

    def critical(self, message: str, **extra):
        """Log critical message with context."""
        self.logger.critical(message, extra=self._add_context(extra))


# Global logger instance
logger = StructuredLogger("code-explainer", os.environ.get("LOG_LEVEL", "INFO"))
