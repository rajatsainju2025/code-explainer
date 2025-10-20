"""
Performance profiling utilities.
"""

from typing import Dict, Any, Callable, Optional
import time
import functools
import psutil
import os


class PerformanceProfiler:
    """Performance profiler for code explanation."""

    def __init__(self):
        self.metrics = {}
        self._process = psutil.Process(os.getpid())

    def profile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function execution - main method expected by tests."""
        return self.profile_function(func, *args, **kwargs)

    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function execution."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = self._get_memory_usage()

            return {
                "execution_time": end_time - start_time,
                "memory_usage_start": start_memory,
                "memory_usage_end": end_memory,
                "memory_delta": end_memory - start_memory,
                "success": True,
                "result": result
            }
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            return {
                "execution_time": end_time - start_time,
                "memory_usage_start": start_memory,
                "memory_usage_end": end_memory,
                "memory_delta": end_memory - start_memory,
                "success": False,
                "error": str(e)
            }

    def start_profiling(self, name: str):
        """Start profiling a section."""
        self.metrics[name] = {
            "start_time": time.time(),
            "start_memory": self._get_memory_usage()
        }

    def end_profiling(self, name: str) -> Dict[str, Any]:
        """End profiling and return metrics."""
        if name in self.metrics:
            start_time = self.metrics[name]["start_time"]
            start_memory = self.metrics[name]["start_memory"]

            end_time = time.time()
            end_memory = self._get_memory_usage()

            result = {
                "execution_time": end_time - start_time,
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
        times = []
        memories = []

        for _ in range(iterations):
            start_time = time.time()
            start_memory = self._get_memory_usage()

            try:
                operation(*args, **kwargs)
            except Exception:
                pass  # We just want performance metrics

            end_time = time.time()
            end_memory = self._get_memory_usage()

            times.append(end_time - start_time)
            memories.append(end_memory - start_memory)

        return {
            "iterations": iterations,
            "avg_execution_time": sum(times) / len(times),
            "min_execution_time": min(times),
            "max_execution_time": max(times),
            "avg_memory_delta": sum(memories) / len(memories),
            "total_time": sum(times)
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
            return self._process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0