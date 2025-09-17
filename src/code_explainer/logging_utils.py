"""Advanced logging configuration and utilities."""

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional, Dict
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rich_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """Setup advanced logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        rich_console: Use Rich for console logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    # Clear existing handlers
    logger = logging.getLogger()
    logger.handlers.clear()
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        "%(levelname)s - %(message)s"
    )
    
    # Console handler
    if rich_console:
        console = Console()
        console_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            rich_tracebacks=True
        )
        console_handler.setFormatter(simple_formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)


class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self, name: str = "performance"):
        """Initialize the performance logger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self._timers: Dict[str, float] = {}
        self._memory_cache: Optional[float] = None
        self._memory_cache_time: float = 0.0
        self._cache_ttl: float = 1.0  # Cache memory readings for 1 second
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation.
        
        Args:
            operation: Name of the operation being timed
        """
        self._timers[operation] = time.time()
    
    def end_timer(self, operation: str, extra_info: Optional[str] = None) -> float:
        """End timing an operation and log the duration.
        
        Args:
            operation: Name of the operation that was timed
            extra_info: Additional information to log
            
        Returns:
            Duration in seconds
        """
        if operation not in self._timers:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return 0.0
        
        duration = time.time() - self._timers[operation]
        del self._timers[operation]
        
        info_str = f" - {extra_info}" if extra_info else ""
        self.logger.info(f"{operation}: {duration:.3f}s{info_str}")
        
        return duration
    
    def log_memory_usage(self, operation: str) -> None:
        """Log current memory usage with caching for performance.

        Args:
            operation: Description of when memory is being measured
        """
        if not HAS_PSUTIL or psutil is None:
            self.logger.debug("psutil not available for memory monitoring")
            return

        try:
            current_time = time.time()
            # Use cached memory reading if recent
            if (self._memory_cache is not None and
                current_time - self._memory_cache_time < self._cache_ttl):
                memory_mb = self._memory_cache
            else:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self._memory_cache = memory_mb
                self._memory_cache_time = current_time

            self.logger.info(f"Memory usage after {operation}: {memory_mb:.1f} MB")
        except Exception as e:
            self.logger.debug(f"Failed to get memory usage: {e}")
    
    def log_model_info(self, model_name: str, num_parameters: Optional[int] = None) -> None:
        """Log model information.
        
        Args:
            model_name: Name of the model
            num_parameters: Number of parameters in the model
        """
        info = f"Loaded model: {model_name}"
        if num_parameters:
            info += f" ({num_parameters:,} parameters)"
        self.logger.info(info)


@lru_cache(maxsize=128)
def get_logger(name: str) -> logging.Logger:
    """Get a cached logger with the given name.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    return logging.getLogger(name)
