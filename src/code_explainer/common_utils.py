"""Common utilities and patterns used across the codebase."""

import functools
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import threading


class SingletonMeta(type):
    """Thread-safe singleton metaclass."""
    _instances: Dict[type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigManager(metaclass=SingletonMeta):
    """Singleton configuration manager to avoid loading config multiple times."""

    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._config_times: Dict[str, float] = {}

    def get_config(self, config_path: Union[str, Path], force_reload: bool = False) -> Dict[str, Any]:
        """Get configuration with caching."""
        path_str = str(config_path)

        # Check if we need to reload
        if not force_reload and path_str in self._configs:
            config_file = Path(config_path)
            if config_file.exists():
                mtime = config_file.stat().st_mtime
                if mtime <= self._config_times.get(path_str, 0):
                    return self._configs[path_str]

        # Load and cache config
        from .utils import load_config as _load_config
        config = _load_config(config_path)
        self._configs[path_str] = config
        self._config_times[path_str] = time.time()

        return config


class LoggingManager(metaclass=SingletonMeta):
    """Singleton logging manager for consistent logging setup."""

    def __init__(self):
        self._configured = False
        self._lock = threading.Lock()

    def setup_logging(
        self,
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        rich_console: bool = True
    ) -> None:
        """Setup logging with consistent configuration."""
        with self._lock:
            if self._configured:
                return

            # Import here to avoid circular imports
            from .logging_utils import setup_logging as _setup_logging
            _setup_logging(
                level=level,
                log_file=log_file,
                rich_console=rich_console,
                max_bytes=max_bytes,
                backup_count=backup_count
            )
            self._configured = True


class TimerManager:
    """Manager for timing operations with automatic cleanup."""

    def __init__(self):
        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()

    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            with self._lock:
                if operation not in self._timers:
                    self._timers[operation] = 0
                self._timers[operation] += duration

            logger = logging.getLogger(__name__)
            logger.debug(f"Operation '{operation}' took {duration:.3f}s")

    def get_timer_stats(self) -> Dict[str, float]:
        """Get timer statistics."""
        with self._lock:
            return self._timers.copy()

    def reset_timer(self, operation: str) -> None:
        """Reset timer for an operation."""
        with self._lock:
            self._timers.pop(operation, None)


def cached_property(func: Callable) -> property:
    """Thread-safe cached property decorator."""
    @functools.wraps(func)
    def wrapper(self):
        cache_key = f"_cached_{func.__name__}"
        if not hasattr(self, cache_key):
            with threading.Lock():
                if not hasattr(self, cache_key):
                    setattr(self, cache_key, func(self))
        return getattr(self, cache_key)
    return property(wrapper)


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying operations on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger = logging.getLogger(__name__)
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"Function {func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator


class ResourceManager:
    """Context manager for resource management with automatic cleanup."""

    def __init__(self):
        self._resources: List[Callable] = []
        self._lock = threading.Lock()

    def add_resource(self, cleanup_func: Callable) -> None:
        """Add a resource cleanup function."""
        with self._lock:
            self._resources.append(cleanup_func)

    @contextmanager
    def manage_resource(self, resource, cleanup_func: Callable):
        """Context manager for managing a single resource."""
        try:
            yield resource
        finally:
            try:
                cleanup_func(resource)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Error cleaning up resource: {e}")

    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        with self._lock:
            for cleanup_func in reversed(self._resources):
                try:
                    cleanup_func()
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in resource cleanup: {e}")
            self._resources.clear()


# Global instances
config_manager = ConfigManager()
logging_manager = LoggingManager()
timer_manager = TimerManager()
resource_manager = ResourceManager()


def get_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Get configuration using the singleton manager."""
    return config_manager.get_config(config_path)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    rich_console: bool = True
) -> None:
    """Setup logging using the singleton manager."""
    logging_manager.setup_logging(level, log_file, max_bytes, backup_count, rich_console)


# Convenience functions for common patterns
def with_timer(operation: str):
    """Decorator to time a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer_manager.timer(operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_function_call(func: Callable) -> Callable:
    """Decorator to log function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    return wrapper