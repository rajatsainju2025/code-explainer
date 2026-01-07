"""Performance metrics collection utilities for optimization tracking."""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Metrics:
    """Container for performance metrics."""
    name: str
    total_time: float = 0.0
    count: int = 0
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    times: List[float] = field(default_factory=list)
    
    def add(self, elapsed: float):
        """Add a measurement."""
        self.total_time += elapsed
        self.count += 1
        self.min_time = min(self.min_time or elapsed, elapsed)
        self.max_time = max(self.max_time or elapsed, elapsed)
        # Keep last 1000 samples for efficiency
        if len(self.times) < 1000:
            self.times.append(elapsed)
    
    @property
    def avg_time(self) -> float:
        """Get average time."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def p95(self) -> Optional[float]:
        """Get 95th percentile time."""
        if not self.times:
            return None
        sorted_times = sorted(self.times)
        idx = int(0.95 * len(sorted_times))
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    @property
    def p99(self) -> Optional[float]:
        """Get 99th percentile time."""
        if not self.times:
            return None
        sorted_times = sorted(self.times)
        idx = int(0.99 * len(sorted_times))
        return sorted_times[min(idx, len(sorted_times) - 1)]


class PerformanceMonitor:
    """Monitor performance metrics across the system."""
    
    __slots__ = ('metrics',)
    
    def __init__(self):
        """Initialize monitor."""
        self.metrics: Dict[str, Metrics] = {}
    
    @contextmanager
    def measure(self, name: str):
        """Context manager to measure execution time.
        
        Usage:
            with monitor.measure("my_operation"):
                # code to measure
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.record(name, elapsed)
    
    def record(self, name: str, elapsed: float):
        """Record a measurement."""
        if name not in self.metrics:
            self.metrics[name] = Metrics(name=name)
        self.metrics[name].add(elapsed)
    
    def get_stats(self, name: str) -> Optional[Dict]:
        """Get statistics for a metric."""
        if name not in self.metrics:
            return None
        
        m = self.metrics[name]
        return {
            "count": m.count,
            "total": m.total_time,
            "avg": m.avg_time,
            "min": m.min_time,
            "max": m.max_time,
            "p95": m.p95,
            "p99": m.p99,
        }
    
    def report(self) -> Dict:
        """Get report of all metrics."""
        return {
            name: self.get_stats(name)
            for name in self.metrics
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def time_function(func: Callable) -> Callable:
    """Decorator to measure function execution time.
    
    Usage:
        @time_function
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        monitor = get_monitor()
        with monitor.measure(func.__name__):
            return func(*args, **kwargs)
    return wrapper
