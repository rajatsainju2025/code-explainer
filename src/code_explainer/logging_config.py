"""Structured logging configuration for production.

Creates console + optional file handlers. File logging only activates when
LOG_DIR is explicitly set or writable — never crashes on permission errors.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional
import os


def _try_import_json_formatter():
    """Lazy import of JSON formatter — falls back to standard formatter."""
    try:
        from pythonjsonlogger import jsonlogger
        return jsonlogger.JsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    except ImportError:
        return logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


class StructuredLogger:
    """Structured JSON logger for production deployments.

    File handlers are only attached when the log directory is writable.
    This avoids PermissionError on systems without /var/log access.
    """

    __slots__ = ('logger',)

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()

        formatter = _try_import_json_formatter()

        # Console handler (STDOUT) — always attached
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handlers — only if directory is writable
        log_dir = os.environ.get("LOG_DIR", "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except (OSError, PermissionError):
            return

        # os.access is lighter than creating a probe file and avoids leaving
        # a stale .write_test artefact if the process is killed mid-check.
        if not os.access(log_dir, os.W_OK):
            return

        try:
            log_file = os.path.join(log_dir, "app.log")
            file_handler = RotatingFileHandler(
                log_file, maxBytes=50 * 1024 * 1024, backupCount=5  # 50 MB
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            error_log_file = os.path.join(log_dir, "error.log")
            error_handler = RotatingFileHandler(
                error_log_file, maxBytes=50 * 1024 * 1024, backupCount=5
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)
        except (OSError, PermissionError):
            pass  # Gracefully degrade to console-only

    def _add_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
