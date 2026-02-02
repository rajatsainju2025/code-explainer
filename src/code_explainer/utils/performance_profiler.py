"""High-performance profiling decorator for bottleneck identification.

This module provides lightweight profiling decorators with minimal overhead
for production use.

Optimized with:
- Conditional profiling with environment variable control
- Minimal overhead when disabled (< 1% performance impact)
- Aggregated statistics to reduce logging overhead
- Thread-safe profiling data collection
"""

import functools
import os
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, TypeVar

# Check if profiling is enabled
_PROFILING_ENABLED = os.getenv("CODE_EXPLAINER_PROFILING", "0") == "1"

# Global profiling statistics
_PROFILE_STATS: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {
        "call_count": 0,
        "total_time": 0.0,
        "min_time": float("inf"),
        "max_time": 0.0,
        "avg_time": 0.0,
    }
)
_STATS_LOCK = threading.RLock()

T = TypeVar("T")


def profile_function(func: Callable[..., T]) -> Callable[..., T]:
    """Lightweight profiling decorator with minimal overhead.
    
    Usage:
        @profile_function
        def my_function(x, y):
            return x + y
    
    Enable profiling:
        export CODE_EXPLAINER_PROFILING=1
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    if not _PROFILING_ENABLED:
        # Fast path: no profiling overhead when disabled
        return func
    
    func_name = f"{func.__module__}.{func.__qualname__}"
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            _update_stats(func_name, elapsed)
    
    return wrapper


def _update_stats(func_name: str, elapsed: float) -> None:
    """Update profiling statistics for a function call."""
    with _STATS_LOCK:
        stats = _PROFILE_STATS[func_name]
        stats["call_count"] += 1
        stats["total_time"] += elapsed
        stats["min_time"] = min(stats["min_time"], elapsed)
        stats["max_time"] = max(stats["max_time"], elapsed)
        stats["avg_time"] = stats["total_time"] / stats["call_count"]


def get_profile_stats() -> Dict[str, Dict[str, Any]]:
    """Get all profiling statistics.
    
    Returns:
        Dictionary mapping function names to their statistics
    """
    with _STATS_LOCK:
        return {
            name: dict(stats) for name, stats in _PROFILE_STATS.items()
        }


def reset_profile_stats() -> None:
    """Reset all profiling statistics."""
    with _STATS_LOCK:
        _PROFILE_STATS.clear()


def print_profile_report(min_calls: int = 10, top_n: int = 20) -> None:
    """Print a formatted profiling report.
    
    Args:
        min_calls: Minimum number of calls to include in report
        top_n: Number of top functions to show (by total time)
    """
    if not _PROFILING_ENABLED:
        print("Profiling is disabled. Set CODE_EXPLAINER_PROFILING=1 to enable.")
        return
    
    stats = get_profile_stats()
    
    # Filter and sort by total time
    filtered_stats = {
        name: data
        for name, data in stats.items()
        if data["call_count"] >= min_calls
    }
    
    if not filtered_stats:
        print("No profiling data available.")
        return
    
    sorted_stats = sorted(
        filtered_stats.items(),
        key=lambda x: x[1]["total_time"],
        reverse=True
    )[:top_n]
    
    print("\n" + "=" * 100)
    print(f"PERFORMANCE PROFILING REPORT (Top {top_n} functions)")
    print("=" * 100)
    print(
        f"{'Function':<60} {'Calls':>8} {'Total(s)':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
    )
    print("-" * 100)
    
    for func_name, data in sorted_stats:
        # Truncate long function names
        display_name = func_name if len(func_name) <= 60 else func_name[:57] + "..."
        print(
            f"{display_name:<60} "
            f"{data['call_count']:>8} "
            f"{data['total_time']:>10.3f} "
            f"{data['avg_time'] * 1000:>10.3f} "
            f"{data['min_time'] * 1000:>10.3f} "
            f"{data['max_time'] * 1000:>10.3f}"
        )
    
    print("=" * 100)
    print(f"Total functions profiled: {len(stats)}")
    print(f"Total function calls: {sum(s['call_count'] for s in stats.values())}")
    print("=" * 100)


def profile_context(name: str):
    """Context manager for profiling code blocks.
    
    Usage:
        with profile_context("data_loading"):
            data = load_data()
    
    Args:
        name: Name for the profiled code block
    """
    class ProfileContext:
        def __init__(self, context_name: str):
            self.context_name = context_name
            self.start_time = 0.0
        
        def __enter__(self):
            if _PROFILING_ENABLED:
                self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if _PROFILING_ENABLED:
                elapsed = time.perf_counter() - self.start_time
                _update_stats(self.context_name, elapsed)
    
    return ProfileContext(name)


# Export public API
__all__ = [
    "profile_function",
    "profile_context",
    "get_profile_stats",
    "reset_profile_stats",
    "print_profile_report",
]
