"""Performance monitoring and profiling utilities for optimization tracking.

This module provides tools to measure and track performance improvements
across the codebase without impacting runtime efficiency.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque


@dataclass
class TimingMetric:
    """Single timing measurement."""
    operation: str
    duration: float
    timestamp: float
    success: bool


class PerformanceMonitor:
    """Monitors performance metrics with minimal overhead."""
    
    __slots__ = ('_metrics', '_lock', '_max_entries')
    
    def __init__(self, max_entries: int = 10000):
        """Initialize monitor."""
        self._metrics: deque = deque(maxlen=max_entries)
        self._lock = threading.RLock()
        self._max_entries = max_entries
    
    def record_timing(self, operation: str, duration: float, success: bool = True) -> None:
        """Record timing for operation."""
        with self._lock:
            metric = TimingMetric(
                operation=operation,
                duration=duration,
                timestamp=time.time(),
                success=success
            )
            self._metrics.append(metric)
    
    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get timing statistics."""
        with self._lock:
            if operation:
                metrics = [m for m in self._metrics if m.operation == operation]
            else:
                metrics = list(self._metrics)
        
        if not metrics:
            return {}
        
        durations = [m.duration for m in metrics]
        return {
            'operation': operation or 'all',
            'count': len(metrics),
            'min': min(durations),
            'max': max(durations),
            'avg': sum(durations) / len(durations),
            'p50': sorted(durations)[len(durations) // 2],
            'p95': sorted(durations)[int(len(durations) * 0.95)],
            'p99': sorted(durations)[int(len(durations) * 0.99)],
        }
    
    def clear(self) -> None:
        """Clear metrics."""
        with self._lock:
            self._metrics.clear()


class FunctionProfiler:
    """Profiles function execution times."""
    
    __slots__ = ('_profiles', '_lock')
    
    def __init__(self):
        """Initialize profiler."""
        self._profiles: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def profile(self, fn: Callable) -> Callable:
        """Decorator to profile function."""
        fn_name = fn.__name__
        
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                duration = time.time() - start
                
                with self._lock:
                    if fn_name not in self._profiles:
                        self._profiles[fn_name] = []
                    self._profiles[fn_name].append(duration)
                
                return result
            except Exception as e:
                raise
        
        return wrapper
    
    def get_profile_stats(self, fn_name: str) -> Optional[Dict[str, float]]:
        """Get profiling statistics."""
        with self._lock:
            if fn_name not in self._profiles:
                return None
            
            times = self._profiles[fn_name]
            if not times:
                return None
            
            return {
                'calls': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
            }
    
    def get_slowest_functions(self, limit: int = 10) -> List[tuple]:
        """Get slowest functions by total time."""
        with self._lock:
            stats = []
            for fn_name, times in self._profiles.items():
                if times:
                    stats.append((fn_name, sum(times)))
            
            return sorted(stats, key=lambda x: x[1], reverse=True)[:limit]


class ResourceMonitor:
    """Monitors resource usage (memory, CPU, etc)."""
    
    __slots__ = ('_samples', '_lock')
    
    def __init__(self):
        """Initialize resource monitor."""
        self._samples: List[Dict[str, float]] = []
        self._lock = threading.RLock()
    
    def sample_memory(self) -> float:
        """Sample current memory usage."""
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024  # MB
    
    def record_sample(self) -> None:
        """Record resource sample."""
        try:
            sample = {
                'timestamp': time.time(),
                'memory_mb': self.sample_memory(),
            }
            
            with self._lock:
                self._samples.append(sample)
                # Keep last 1000 samples
                if len(self._samples) > 1000:
                    self._samples.pop(0)
        except Exception:
            pass
    
    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """Get memory statistics."""
        with self._lock:
            if not self._samples:
                return None
            
            memories = [s['memory_mb'] for s in self._samples]
            return {
                'current': memories[-1],
                'min': min(memories),
                'max': max(memories),
                'avg': sum(memories) / len(memories),
            }


# Global instances
_performance_monitor = PerformanceMonitor()
_function_profiler = FunctionProfiler()
_resource_monitor = ResourceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    return _performance_monitor


def get_function_profiler() -> FunctionProfiler:
    """Get global function profiler."""
    return _function_profiler


def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor."""
    return _resource_monitor


def profile_function(fn: Callable) -> Callable:
    """Profile a function."""
    return _function_profiler.profile(fn)


def record_timing(operation: str, duration: float) -> None:
    """Record timing for operation."""
    _performance_monitor.record_timing(operation, duration)
