"""Centralized error handling and custom exceptions."""

from typing import Optional, Dict, Any
import traceback
from logging_config import logger


class CodeExplainerException(Exception):
    """Base exception for Code Explainer."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize exception.

        Args:
            message: Human-readable error message
            code: Machine-readable error code
            status_code: HTTP status code
            details: Additional error context
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "error": {
                "message": self.message,
                "code": self.code,
                "details": self.details
            }
        }


class ValidationError(CodeExplainerException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str = None):
        super().__init__(
            message,
            code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field} if field else {}
        )


class ModelError(CodeExplainerException):
    """Raised when model inference fails."""

    def __init__(self, message: str, model_name: str = None):
        super().__init__(
            message,
            code="MODEL_ERROR",
            status_code=500,
            details={"model": model_name} if model_name else {}
        )


class DatabaseError(CodeExplainerException):
    """Raised when database operation fails."""

    def __init__(self, message: str, operation: str = None):
        super().__init__(
            message,
            code="DATABASE_ERROR",
            status_code=500,
            details={"operation": operation} if operation else {}
        )


class CacheError(CodeExplainerException):
    """Raised when cache operation fails."""

    def __init__(self, message: str, cache_type: str = None):
        super().__init__(
            message,
            code="CACHE_ERROR",
            status_code=500,
            details={"cache_type": cache_type} if cache_type else {}
        )


def handle_exception(exc: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Centralized exception handler.

    Args:
        exc: Exception to handle
        context: Additional context (request path, user, etc.)

    Returns:
        Structured error response
    """
    context = context or {}

    if isinstance(exc, CodeExplainerException):
        logger.warning(
            f"{exc.code}: {exc.message}",
            code=exc.code,
            status_code=exc.status_code,
            **context
        )
        return {
            "error": {
                "message": exc.message,
                "code": exc.code,
                "status_code": exc.status_code,
                "details": exc.details
            }
        }
    else:
        # Log unexpected exceptions with full traceback
        logger.error(
            f"Unexpected error: {str(exc)}",
            exc_info=True,
            error_type=type(exc).__name__,
            **context
        )
        return {
            "error": {
                "message": "Internal server error",
                "code": "INTERNAL_ERROR",
                "status_code": 500
            }
        }


class ErrorContext:
    """Context manager for error handling with context."""

    def __init__(self, operation: str, **context):
        """Initialize error context.

        Args:
            operation: Operation name for logging
            **context: Additional context fields
        """
        self.operation = operation
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            context = {**self.context, "operation": self.operation}
            handle_exception(exc_val, context)
            return False
        return True
