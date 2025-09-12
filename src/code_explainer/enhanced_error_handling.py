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
"""

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

# Custom exception hierarchy
class CodeExplainerError(Exception):
    """Base exception for Code Explainer."""
    pass

class ModelError(CodeExplainerError):
    """Exception raised for model-related errors."""
    pass

class ConfigurationError(CodeExplainerError):
    """Exception raised for configuration-related errors."""
    pass

class ValidationError(CodeExplainerError):
    """Exception raised for input validation errors."""
    pass

class ProcessingError(CodeExplainerError):
    """Exception raised for processing-related errors."""
    pass

class ResourceError(CodeExplainerError):
    """Exception raised for resource-related errors."""
    pass

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    message: str
    module: str
    function: str
    line_number: int
    exception_info: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""

    def __init__(self, name: str, log_level: str = "INFO",
                 log_file: Optional[Path] = None,
                 console_output: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Structured log storage
        self.log_entries: List[LogEntry] = []
        self.max_entries = 10000

    def _create_log_entry(self, level: str, message: str,
                         extra_data: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Create a structured log entry."""
        frame = sys._getframe(2)
        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            module=frame.f_code.co_filename,
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            extra_data=extra_data or {}
        )

    def _store_entry(self, entry: LogEntry):
        """Store log entry in memory."""
        self.log_entries.append(entry)
        if len(self.log_entries) > self.max_entries:
            self.log_entries = self.log_entries[-self.max_entries:]

    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log info message."""
        entry = self._create_log_entry("INFO", message, extra_data)
        self._store_entry(entry)
        self.logger.info(message, extra=extra_data)

    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        entry = self._create_log_entry("WARNING", message, extra_data)
        self._store_entry(entry)
        self.logger.warning(message, extra=extra_data)

    def error(self, message: str, exc_info: bool = True,
              extra_data: Optional[Dict[str, Any]] = None):
        """Log error message."""
        entry = self._create_log_entry("ERROR", message, extra_data)
        if exc_info:
            entry.exception_info = traceback.format_exc()
        self._store_entry(entry)
        self.logger.error(message, exc_info=exc_info, extra=extra_data)

    def critical(self, message: str, exc_info: bool = True,
                 extra_data: Optional[Dict[str, Any]] = None):
        """Log critical message."""
        entry = self._create_log_entry("CRITICAL", message, extra_data)
        if exc_info:
            entry.exception_info = traceback.format_exc()
        self._store_entry(entry)
        self.logger.critical(message, exc_info=exc_info, extra=extra_data)

    def get_recent_logs(self, hours: int = 24) -> List[LogEntry]:
        """Get recent log entries."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [entry for entry in self.log_entries
                if entry.timestamp > cutoff]

    def export_logs(self, filepath: Path, format: str = "json"):
        """Export logs to file."""
        if format == "json":
            data = [entry.__dict__ for entry in self.log_entries]
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "csv":
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=LogEntry.__dataclass_fields__.keys())
                writer.writeheader()
                for entry in self.log_entries:
                    writer.writerow(entry.__dict__)

class ErrorHandler:
    """Centralized error handling and recovery."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_counts: Dict[str, int] = {}
        self.max_retries = 3
        self.backoff_factor = 2.0

    def register_recovery_strategy(self, error_type: str,
                                  strategy: Callable):
        """Register a recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy

    def handle_error(self, error: Exception, context: str = "",
                    retry_func: Optional[Callable] = None) -> Any:
        """Handle an error with appropriate recovery."""
        error_type = type(error).__name__

        # Log the error
        self.logger.error(
            f"Error in {context}: {str(error)}",
            extra_data={"error_type": error_type, "context": context}
        )

        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Try recovery strategy
        if error_type in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {error_type}")
                return self.recovery_strategies[error_type](error)
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {str(recovery_error)}")

        # Try retry if function provided
        if retry_func:
            return self._retry_with_backoff(retry_func, error_type)

        # Re-raise if no recovery possible
        raise error

    def _retry_with_backoff(self, func: Callable, error_type: str) -> Any:
        """Retry function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func()
            except Exception as e:
                wait_time = self.backoff_factor ** attempt
                self.logger.warning(
                    f"Retry {attempt + 1}/{self.max_retries} failed: {str(e)}. "
                    f"Waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)

        raise Exception(f"Max retries exceeded for {error_type}")

class PerformanceMonitor:
    """Monitor system performance and detect anomalies."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.metrics: Dict[str, List[float]] = {}
        self.alerts: Queue = Queue()
        self.thresholds = {
            "response_time": 5.0,  # seconds
            "memory_usage": 0.8,   # 80%
            "cpu_usage": 0.9,      # 90%
            "error_rate": 0.1      # 10%
        }
        self.monitoring_active = False

    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False

    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

        # Keep only recent metrics
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

        # Check thresholds
        self._check_thresholds(name, value)

    def _check_thresholds(self, name: str, value: float):
        """Check if metric exceeds threshold."""
        if name in self.thresholds and value > self.thresholds[name]:
            alert_msg = f"Threshold exceeded for {name}: {value} > {self.thresholds[name]}"
            self.logger.warning(alert_msg)
            self.alerts.put({
                "timestamp": datetime.now(),
                "metric": name,
                "value": value,
                "threshold": self.thresholds[name],
                "message": alert_msg
            })

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                import psutil
                self.record_metric("cpu_usage", psutil.cpu_percent() / 100.0)
                self.record_metric("memory_usage", psutil.virtual_memory().percent / 100.0)

                # Check for alerts
                while not self.alerts.empty():
                    alert = self.alerts.get()
                    self._handle_alert(alert)

                time.sleep(60)  # Monitor every minute

            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                time.sleep(60)

    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle performance alert."""
        # In a real system, this might send notifications
        self.logger.warning(f"Performance alert: {alert['message']}")

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of collected metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "current": values[-1],
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        return summary

# Decorators for error handling and monitoring
def with_error_handling(logger: StructuredLogger, context: str = ""):
    """Decorator to add error handling to functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler(logger)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return error_handler.handle_error(e, context or func.__name__)
        return wrapper
    return decorator

def with_performance_monitoring(monitor: PerformanceMonitor, metric_name: str):
    """Decorator to monitor function performance."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                monitor.record_metric(f"{metric_name}_time", execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                monitor.record_metric(f"{metric_name}_error_time", execution_time)
                raise e
        return wrapper
    return decorator

# Global instances
logger = StructuredLogger("code_explainer", log_file=Path("logs/code_explainer.log"))
error_handler = ErrorHandler(logger)
performance_monitor = PerformanceMonitor(logger)

# Convenience functions
def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup global logging configuration."""
    global logger
    logger = StructuredLogger("code_explainer", log_level, log_file)

def get_logger() -> StructuredLogger:
    """Get the global logger instance."""
    return logger

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return error_handler

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor

if __name__ == "__main__":
    # Example usage
    setup_logging("DEBUG", Path("logs/example.log"))

    # Start monitoring
    performance_monitor.start_monitoring()

    @with_error_handling(logger, "example_function")
    @with_performance_monitoring(performance_monitor, "example")
    def example_function():
        """Example function with error handling and monitoring."""
        logger.info("Starting example function")
        # Simulate some work
        time.sleep(0.1)
        logger.info("Example function completed")
        return "success"

    try:
        result = example_function()
        print(f"Result: {result}")
    except Exception as e:
        logger.error(f"Function failed: {str(e)}")

    # Export logs
    logger.export_logs(Path("logs/exported_logs.json"))

    # Get metrics summary
    summary = performance_monitor.get_metrics_summary()
    print(f"Metrics summary: {json.dumps(summary, indent=2)}")

    # Stop monitoring
    performance_monitor.stop_monitoring()
