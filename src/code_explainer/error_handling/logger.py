"""Structured logging functionality."""

import json
import logging
import logging.handlers
import sys
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

# Cache datetime.now for faster access
_now = datetime.now
_format_exc = traceback.format_exc


@dataclass(slots=True)
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
            # Ensure Path type
            log_path = log_file if isinstance(log_file, Path) else Path(str(log_file))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Structured log storage - use deque for O(1) append and automatic trimming
        self.log_entries: Deque[LogEntry] = deque(maxlen=10000)

    def _create_log_entry(self, level: str, message: str,
                         extra_data: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Create a structured log entry."""
        frame = sys._getframe(2)
        code = frame.f_code
        return LogEntry(
            timestamp=_now(),
            level=level,
            message=message,
            module=code.co_filename,
            function=code.co_name,
            line_number=frame.f_lineno,
            extra_data=extra_data or {}
        )

    def _store_entry(self, entry: LogEntry):
        """Store log entry in memory."""
        self.log_entries.append(entry)  # deque with maxlen auto-trims

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

    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        entry = self._create_log_entry("DEBUG", message, extra_data)
        self._store_entry(entry)
        self.logger.debug(message, extra=extra_data)

    def error(self, message: str, exc_info: bool = True,
              extra_data: Optional[Dict[str, Any]] = None):
        """Log error message."""
        entry = self._create_log_entry("ERROR", message, extra_data)
        if exc_info:
            entry.exception_info = _format_exc()
        self._store_entry(entry)
        self.logger.error(message, exc_info=exc_info, extra=extra_data)

    def critical(self, message: str, exc_info: bool = True,
                 extra_data: Optional[Dict[str, Any]] = None):
        """Log critical message."""
        entry = self._create_log_entry("CRITICAL", message, extra_data)
        if exc_info:
            entry.exception_info = _format_exc()
        self._store_entry(entry)
        self.logger.critical(message, exc_info=exc_info, extra=extra_data)

    def get_recent_logs(self, hours: int = 24) -> List[LogEntry]:
        """Get recent log entries."""
        cutoff = _now() - timedelta(hours=hours)
        return [entry for entry in self.log_entries
                if entry.timestamp > cutoff]

    def export_logs(self, filepath: Path, format: str = "json"):
        """Export logs to file."""
        if format == "json":
            data = [entry.__dict__ for entry in self.log_entries]
            with open(filepath, 'w') as f:
                json.dump(data, f, separators=(',', ':'), default=str)  # Compact JSON
        elif format == "csv":
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=LogEntry.__dataclass_fields__.keys())
                writer.writeheader()
                for entry in self.log_entries:
                    writer.writerow(entry.__dict__)