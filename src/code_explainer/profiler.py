"""
Performance profiling utilities.

Optimized for:
- Minimal allocation overhead during profiling
- Pre-allocated result dictionaries
- Context manager support for cleaner code
- Lazy psutil initialization for faster startup
- Async-aware profiling support
"""

import asyncio
from typing import Dict, Any, Callable, Optional, List
import time
import functools
from contextlib import contextmanager, asynccontextmanager
import os

# Use perf_counter for high-precision timing
_perf_counter = time.perf_counter

# Lazy psutil import to speed up module loading
_process = None

def _get_process():
    """Lazily initialize psutil process handle."""
    global _process
    if _process is None:
        try:
            import psutil
            _process = psutil.Process(os.getpid())
        except ImportError:
            _process = False  # Mark as unavailable
    return _process


# Pre-computed divisor to avoid repeated division
_MEMORY_MB_DIVISOR = 1024 * 1024

# Pre-allocated result templates to reduce allocation
_PROFILE_RESULT_KEYS = frozenset({
    "execution_time", "memory_usage_start", "memory_usage_end", 
    "memory_delta", "success", "result", "error"
})


class PerformanceProfiler:
    """Performance profiler for code explanation.
    
    Uses __slots__ for memory efficiency and lazy process initialization.
    """
    
    __slots__ = ('metrics', '_active_contexts')

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._active_contexts: Dict[str, float] = {}  # name -> start_time

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = _get_process()
        if process is False:
            return 0.0
        try:
            return process.memory_info().rss / _MEMORY_MB_DIVISOR
        except Exception:
            return 0.0

    @contextmanager
    def profile(self, name: str = "default"):
        """Context manager for profiling code blocks.
        
        Usage:
            with profiler.profile("my_operation") as metrics:
                # code to profile
            print(metrics)  # Access timing metrics
        """
        start_time = _perf_counter()
        start_memory = self._get_memory_usage()
        result = {}
        
        try:
            yield result
            result["success"] = True
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            raise
        finally:
            end_time = _perf_counter()
            end_memory = self._get_memory_usage()
            
            result["execution_time"] = end_time - start_time
            result["memory_usage_start"] = start_memory
            result["memory_usage_end"] = end_memory
            result["memory_delta"] = end_memory - start_memory
            
            # Store in metrics
            self.metrics[name] = result
    
    @asynccontextmanager
    async def profile_async(self, name: str = "default"):
        """Async context manager for profiling async code blocks.
        
        Usage:
            async with profiler.profile_async("my_operation") as metrics:
                await async_operation()
            print(metrics)  # Access timing metrics
        """
        start_time = _perf_counter()
        start_memory = self._get_memory_usage()
        result = {}
        
        try:
            yield result
            result["success"] = True
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            raise
        finally:
            end_time = _perf_counter()
            end_memory = self._get_memory_usage()
            
            result["execution_time"] = end_time - start_time
            result["memory_usage_start"] = start_memory
            result["memory_usage_end"] = end_memory
            result["memory_delta"] = end_memory - start_memory
            
            # Store in metrics
            self.metrics[name] = result

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

    def start_profiling(self, name: str) -> None:
        """Start profiling a section."""
        self.metrics[name] = {
            "start_time": _perf_counter(),
            "start_memory": self._get_memory_usage()
        }

    def end_profiling(self, name: str) -> Dict[str, Any]:
        """End profiling and return metrics."""
        if name not in self.metrics:
            return {
                "execution_time": 0,
                "memory_usage_start": 0,
                "memory_usage_end": 0,
                "memory_delta": 0,
                "success": False,
                "error": f"No profiling session found for '{name}'"
            }
        
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

    def benchmark_operation(self, operation: Callable, iterations: int = 10, 
                           warmup: int = 2, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark an operation over multiple iterations.
        
        Args:
            operation: Function to benchmark
            iterations: Number of measured iterations
            warmup: Number of warmup iterations (excluded from stats)
        """
        # Warmup runs (not measured)
        for _ in range(warmup):
            try:
                operation(*args, **kwargs)
            except Exception:
                pass
        
        # Pre-allocate lists for efficiency
        times: List[float] = [0.0] * iterations
        memories: List[float] = [0.0] * iterations

        for i in range(iterations):
            start_time = _perf_counter()
            start_memory = self._get_memory_usage()

            try:
                operation(*args, **kwargs)
            except Exception:
                pass  # We just want performance metrics

            times[i] = _perf_counter() - start_time
            memories[i] = self._get_memory_usage() - start_memory

        # Compute statistics
        total_time = sum(times)
        n = iterations
        
        # Sort times for percentile calculation
        sorted_times = sorted(times)
        
        return {
            "iterations": iterations,
            "warmup": warmup,
            "avg_execution_time": total_time / n,
            "min_execution_time": sorted_times[0],
            "max_execution_time": sorted_times[-1],
            "p50_execution_time": sorted_times[n // 2],
            "p95_execution_time": sorted_times[int(n * 0.95)] if n >= 20 else sorted_times[-1],
            "avg_memory_delta": sum(memories) / n,
            "total_time": total_time
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling sessions."""
        return {
            "active_sessions": list(self.metrics.keys()),
            "total_sessions": len(self.metrics)
        }

    def clear(self) -> None:
        """Clear all profiling metrics."""
        self.metrics.clear()

    def save_metrics(self, filepath: str) -> bool:
        """Save metrics to file."""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
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