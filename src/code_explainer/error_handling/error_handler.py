"""Error handling coordination."""

import asyncio
import functools
import threading
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from .exceptions import CodeExplainerError
from .logger import StructuredLogger

T = TypeVar('T')


class ErrorHandler:
    """Centralized error handling and recovery coordinator."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register_recovery_strategy(self, error_type: str,
                                  strategy: Callable):
        """Register a recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type}")

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                    attempt_recovery: bool = True) -> bool:
        """Handle an error with optional recovery."""
        error_type = type(error).__name__
        context = context or {}

        # Increment error count
        with self._lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log the error
        self.logger.error(
            f"Error occurred: {error}",
            extra_data={
                "error_type": error_type,
                "error_count": self.error_counts[error_type],
                "context": context
            }
        )

        # Attempt recovery if requested
        if attempt_recovery and error_type in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {error_type}")
                self.recovery_strategies[error_type](error, context)
                self.logger.info(f"Recovery successful for {error_type}")
                return True
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery failed for {error_type}: {recovery_error}",
                    extra_data={"original_error": str(error)}
                )

        return False

    def wrap_async(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator to wrap async functions with error handling."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                self.handle_error(e, context)
                raise
        return wrapper

    def wrap_sync(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap sync functions with error handling."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                self.handle_error(e, context)
                raise
        return wrapper

    @contextmanager
    def error_context(self, operation: str, **context):
        """Context manager for error handling with operation context."""
        start_time = time.time()
        try:
            self.logger.debug(f"Starting operation: {operation}",
                            extra_data=context)
            yield
            duration = time.time() - start_time
            self.logger.info(f"Operation completed: {operation}",
                           extra_data={"duration": duration, **context})
        except Exception as e:
            duration = time.time() - start_time
            context.update({"duration": duration, "operation": operation})
            self.handle_error(e, context)
            raise

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            return {
                "total_errors": sum(self.error_counts.values()),
                "error_types": dict(self.error_counts),
                "recovery_strategies": list(self.recovery_strategies.keys())
            }

    def reset_error_counts(self):
        """Reset error count statistics."""
        with self._lock:
            self.error_counts.clear()
        self.logger.info("Error counts reset")