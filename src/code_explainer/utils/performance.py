"""Performance monitoring and profiling utilities."""

import time
import functools
from typing import Callable, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, list] = {}
    
    def record_time(self, name: str, seconds: float) -> None:
        """Record timing metric.
        
        Args:
            name: Metric name
            seconds: Time in seconds
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(seconds)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with min, max, avg
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        times = self.metrics[name]
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "count": len(times)
        }


def time_function(func: Callable) -> Callable:
    """Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


# Global performance monitor
_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor.
    
    Returns:
        Global PerformanceMonitor instance
    """
    return _monitor
