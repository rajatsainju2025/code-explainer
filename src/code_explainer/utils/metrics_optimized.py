"""Lock-free and optimized metrics collection for reduced contention.

This module provides efficient metrics collection with minimal lock overhead
using atomic operations and lock-free data structures where possible.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class LockFreeCounter:
    """Atomic counter using thread-safe operations."""
    
    __slots__ = ('_value', '_lock')
    
    _value: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    def increment(self, delta: int = 1) -> int:
        """Increment counter (atomic).
        
        Args:
            delta: Amount to increment
            
        Returns:
            New counter value
        """
        with self._lock:
            self._value += delta
            return self._value
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        """Reset counter."""
        with self._lock:
            self._value = 0


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    timestamp: float
    value: float


class RollingWindow:
    """Rolling window of metric values with O(1) insertion."""
    
    __slots__ = ('_window', '_max_size', '_lock', '_sum', '_count')
    
    def __init__(self, max_size: int = 1000):
        """Initialize rolling window.
        
        Args:
            max_size: Maximum values to keep
        """
        self._window: deque = deque(maxlen=max_size)
        self._max_size = max_size
        self._lock = threading.RLock()
        self._sum = 0.0
        self._count = 0
    
    def add(self, value: float) -> None:
        """Add value to window."""
        with self._lock:
            if len(self._window) == self._max_size:
                # Removing oldest - update sum
                removed = self._window[0]
                self._sum -= removed
            
            self._window.append(value)
            self._sum += value
            self._count += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics for window."""
        with self._lock:
            if not self._window:
                return {
                    'count': 0,
                    'sum': 0,
                    'avg': 0,
                    'min': 0,
                    'max': 0,
                    'p50': 0,
                    'p95': 0,
                    'p99': 0,
                }
            
            values = list(self._window)
            sorted_values = sorted(values)
            
            return {
                'count': len(values),
                'sum': self._sum,
                'avg': self._sum / len(values),
                'min': sorted_values[0],
                'max': sorted_values[-1],
                'p50': sorted_values[len(values) // 2],
                'p95': sorted_values[int(len(values) * 0.95)],
                'p99': sorted_values[int(len(values) * 0.99)],
            }
    
    def clear(self) -> None:
        """Clear window."""
        with self._lock:
            self._window.clear()
            self._sum = 0.0
            self._count = 0


class OptimizedMetricsCollector:
    """Optimized metrics collector with reduced lock contention."""
    
    __slots__ = ('_counters', '_windows', '_lock', '_last_stats_time')
    
    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, LockFreeCounter] = {}
        self._windows: Dict[str, RollingWindow] = {}
        self._lock = threading.RLock()
        self._last_stats_time = time.time()
    
    def register_counter(self, name: str) -> None:
        """Register a counter metric.
        
        Args:
            name: Counter name
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = LockFreeCounter()
    
    def register_window(self, name: str, max_size: int = 1000) -> None:
        """Register a rolling window metric.
        
        Args:
            name: Window name
            max_size: Maximum values to keep
        """
        with self._lock:
            if name not in self._windows:
                self._windows[name] = RollingWindow(max_size)
    
    def increment_counter(self, name: str, delta: int = 1) -> None:
        """Increment a counter.
        
        Args:
            name: Counter name
            delta: Amount to increment
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = LockFreeCounter()
            counter = self._counters[name]
        
        counter.increment(delta)
    
    def record_value(self, name: str, value: float) -> None:
        """Record a value in a rolling window.
        
        Args:
            name: Window name
            value: Value to record
        """
        with self._lock:
            if name not in self._windows:
                self._windows[name] = RollingWindow()
            window = self._windows[name]
        
        window.add(value)
    
    def get_counter_value(self, name: str) -> int:
        """Get counter value.
        
        Args:
            name: Counter name
            
        Returns:
            Counter value or 0 if not found
        """
        with self._lock:
            counter = self._counters.get(name)
        
        return counter.get() if counter else 0
    
    def get_window_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get rolling window statistics.
        
        Args:
            name: Window name
            
        Returns:
            Statistics dict or None if not found
        """
        with self._lock:
            window = self._windows.get(name)
        
        return window.get_stats() if window else None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics snapshot."""
        with self._lock:
            metrics = {
                'counters': {name: counter.get() for name, counter in self._counters.items()},
                'windows': {name: window.get_stats() for name, window in self._windows.items()},
                'timestamp': time.time()
            }
        
        return metrics
    
    def reset_counter(self, name: str) -> None:
        """Reset a counter.
        
        Args:
            name: Counter name
        """
        with self._lock:
            counter = self._counters.get(name)
        
        if counter:
            counter.reset()
    
    def reset_window(self, name: str) -> None:
        """Reset a rolling window.
        
        Args:
            name: Window name
        """
        with self._lock:
            window = self._windows.get(name)
        
        if window:
            window.clear()


class PerformanceBucket:
    """Bucket for collecting performance metrics during a request."""
    
    __slots__ = ('_metrics', '_lock', '_start_time', '_end_time')
    
    def __init__(self):
        """Initialize performance bucket."""
        self._metrics: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._start_time: float = time.time()
        self._end_time: Optional[float] = None
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        with self._lock:
            self._metrics[name] = value
    
    def mark_end(self) -> None:
        """Mark bucket as complete."""
        with self._lock:
            self._end_time = time.time()
    
    def get_duration(self) -> float:
        """Get bucket duration."""
        end = self._end_time or time.time()
        return end - self._start_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics from bucket."""
        with self._lock:
            return {
                **self._metrics,
                'duration': self.get_duration(),
                'start_time': self._start_time,
                'end_time': self._end_time,
            }


# Global singleton
_metrics_collector: Optional[OptimizedMetricsCollector] = None
_collector_lock = threading.RLock()


def get_metrics_collector() -> OptimizedMetricsCollector:
    """Get singleton metrics collector."""
    global _metrics_collector
    
    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = OptimizedMetricsCollector()
                
                # Pre-register common metrics
                _metrics_collector.register_counter('requests_total')
                _metrics_collector.register_counter('requests_success')
                _metrics_collector.register_counter('requests_failed')
                _metrics_collector.register_window('request_duration')
                _metrics_collector.register_window('model_inference_time')
                _metrics_collector.register_window('cache_hit_rate')
    
    return _metrics_collector
