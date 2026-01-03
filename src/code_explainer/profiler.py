"""
Performance profiling utilities.
"""

from typing import Dict, Any, Callable, Optional, List
import time
import functools
import psutil
import os

# Use perf_counter for high-precision timing
_perf_counter = time.perf_counter


class PerformanceProfiler:
    """Performance profiler for code explanation."""
    
    __slots__ = ('metrics', '_process', '_memory_mb_divisor')

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._process = psutil.Process(os.getpid())
        # Pre-compute divisor to avoid repeated division
        self._memory_mb_divisor = 1024 * 1024

    def profile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function execution - main method expected by tests."""
        return self.profile_function(func, *args, **kwargs)

    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function execution."""
        start_time = _perf_counter()
        start_memory = self._get_memory_usage()

        try:
            result = func(*args, **kwargs)
            execution_time = _perf_counter() - start_time
            end_memory = self._get_memory_usage()

            return {
                "execution_time": execution_time,
                "memory_usage_start": start_memory,
                "memory_usage_end": end_memory,
                "memory_delta": end_memory - start_memory,
                "success": True,
                "result": result
            }
        except Exception as e:
            execution_time = _perf_counter() - start_time
            end_memory = self._get_memory_usage()

            return {
                "execution_time": execution_time,
                "memory_usage_start": start_memory,
                "memory_usage_end": end_memory,
                "memory_delta": end_memory - start_memory,
                "success": False,
                "error": str(e)
            }

    def start_profiling(self, name: str):
        """Start profiling a section."""
        self.metrics[name] = {
            "start_time": _perf_counter(),
            "start_memory": self._get_memory_usage()
        }

    def end_profiling(self, name: str) -> Dict[str, Any]:
        """End profiling and return metrics."""
        if name in self.metrics:
            data = self.metrics[name]
            start_time = data["start_time"]
            start_memory = data["start_memory"]

            execution_time = _perf_counter() - start_time
            end_memory = self._get_memory_usage()

            result = {
                "execution_time": execution_time,
                "memory_usage_start": start_memory,
                "memory_usage_end": end_memory,
                "memory_delta": end_memory - start_memory,
                "success": True
            }

            # Clean up
            del self.metrics[name]
            return result

        return {
            "execution_time": 0,
            "memory_usage_start": 0,
            "memory_usage_end": 0,
            "memory_delta": 0,
            "success": False,
            "error": f"No profiling session found for '{name}'"
        }

    def benchmark_operation(self, operation: Callable, iterations: int = 10, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark an operation over multiple iterations."""
        times: List[float] = []
        memories: List[float] = []

        for _ in range(iterations):
            start_time = _perf_counter()
            start_memory = self._get_memory_usage()

            try:
                operation(*args, **kwargs)
            except Exception:
                pass  # We just want performance metrics

            times.append(_perf_counter() - start_time)
            memories.append(self._get_memory_usage() - start_memory)

        # Pre-compute length for reuse
        n = len(times)
        total_time = sum(times)
        
        return {
            "iterations": iterations,
            "avg_execution_time": total_time / n,
            "min_execution_time": min(times),
            "max_execution_time": max(times),
            "avg_memory_delta": sum(memories) / n,
            "total_time": total_time
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling sessions."""
        return {
            "active_sessions": list(self.metrics.keys()),
            "total_sessions": len(self.metrics)
        }

    def save_metrics(self, filepath: str) -> bool:
        """Save metrics to file."""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            return True
        except Exception:
            return False

    def load_metrics(self, filepath: str) -> bool:
        """Load metrics from file."""
        try:
            import json
            with open(filepath, 'r') as f:
                self.metrics = json.load(f)
            return True
        except Exception:
            return False

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self._process.memory_info().rss / self._memory_mb_divisor
        except Exception:
            return 0.0