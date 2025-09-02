"""
Performance Optimization Module for Code Intelligence Platform

This module provides comprehensive performance optimization capabilities
including caching, async processing, memory management, profiling, and
optimization strategies for the code intelligence platform.

Features:
- Multi-level caching system (memory, disk, distributed)
- Async task processing and concurrency management
- Memory optimization and garbage collection
- Performance profiling and bottleneck detection
- Auto-scaling and resource management
- Query optimization for code analysis
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib
import os
import json
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[str] = field(default_factory=list)


class CacheStrategy(ABC):
    """Abstract base class for caching strategies."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class MemoryCache(CacheStrategy):
    """In-memory LRU cache implementation."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from memory cache."""
        if key in self.cache:
            entry = self.cache[key]
            if self._is_expired(entry):
                self.delete(key)
                return None

            # Move to end (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            return entry['value']
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in memory cache."""
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        expiry = time.time() + ttl if ttl else None
        self.cache[key] = {'value': value, 'expiry': expiry}

        if key not in self.access_order:
            self.access_order.append(key)

    def delete(self, key: str) -> None:
        """Delete value from memory cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return entry['expiry'] is not None and time.time() > entry['expiry']


class DiskCache(CacheStrategy):
    """Disk-based cache implementation."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from disk cache."""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    entry = pickle.load(f)

                if self._is_expired(entry):
                    os.remove(cache_path)
                    return None

                return entry['value']
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                return None
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in disk cache."""
        cache_path = self._get_cache_path(key)
        expiry = time.time() + ttl if ttl else None
        entry = {'value': value, 'expiry': expiry}

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.error(f"Failed to store cache entry {key}: {e}")

    def delete(self, key: str) -> None:
        """Delete value from disk cache."""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def clear(self) -> None:
        """Clear all disk cache entries."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                os.remove(os.path.join(self.cache_dir, filename))

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return entry['expiry'] is not None and time.time() > entry['expiry']


class MultiLevelCache:
    """Multi-level caching system combining memory and disk cache."""

    def __init__(self, memory_cache: MemoryCache, disk_cache: DiskCache):
        self.memory_cache = memory_cache
        self.disk_cache = disk_cache

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in both cache levels."""
        self.memory_cache.set(key, value, ttl)
        self.disk_cache.set(key, value, ttl)

    def delete(self, key: str) -> None:
        """Delete from both cache levels."""
        self.memory_cache.delete(key)
        self.disk_cache.delete(key)

    def clear(self) -> None:
        """Clear both cache levels."""
        self.memory_cache.clear()
        self.disk_cache.clear()


class AsyncTaskManager:
    """Manages async tasks and concurrency."""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = asyncio.get_event_loop()
        self.tasks: Dict[str, asyncio.Task] = {}

    async def submit_task(self, task_id: str, coro: Callable) -> Any:
        """Submit an async task."""
        task = asyncio.create_task(coro())
        self.tasks[task_id] = task
        return await task

    def submit_sync_task(self, task_id: str, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Submit a synchronous task to run in thread pool."""
        future = self.loop.run_in_executor(self.executor, func, *args, **kwargs)
        return future

    async def wait_for_task(self, task_id: str) -> Any:
        """Wait for a specific task to complete."""
        if task_id in self.tasks:
            return await self.tasks[task_id]
        raise ValueError(f"Task {task_id} not found")

    def cancel_task(self, task_id: str) -> None:
        """Cancel a running task."""
        if task_id in self.tasks:
            self.tasks[task_id].cancel()
            del self.tasks[task_id]

    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs."""
        return list(self.tasks.keys())


class PerformanceProfiler:
    """Performance profiling and monitoring."""

    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetrics]] = {}
        self.lock = threading.Lock()

    def start_operation(self, operation_name: str) -> str:
        """Start profiling an operation."""
        operation_id = f"{operation_name}_{time.time()}"
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time()
        )

        with self.lock:
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(metrics)

        return operation_id

    def end_operation(self, operation_id: str) -> None:
        """End profiling an operation."""
        # Extract operation name from ID
        operation_name = operation_id.rsplit('_', 1)[0]

        with self.lock:
            if operation_name in self.metrics:
                for metrics in reversed(self.metrics[operation_name]):
                    if not metrics.end_time:
                        metrics.end_time = time.time()
                        metrics.duration = metrics.end_time - metrics.start_time
                        metrics.memory_usage = psutil.virtual_memory().percent
                        metrics.cpu_usage = psutil.cpu_percent()
                        break

    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics."""
        with self.lock:
            if operation_name:
                return {
                    "operation": operation_name,
                    "metrics": [self._metrics_to_dict(m) for m in self.metrics.get(operation_name, [])]
                }
            else:
                return {
                    "all_operations": {
                        op: [self._metrics_to_dict(m) for m in metrics]
                        for op, metrics in self.metrics.items()
                    }
                }

    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation_name": metrics.operation_name,
            "start_time": metrics.start_time,
            "end_time": metrics.end_time,
            "duration": metrics.duration,
            "memory_usage": metrics.memory_usage,
            "cpu_usage": metrics.cpu_usage,
            "cache_hits": metrics.cache_hits,
            "cache_misses": metrics.cache_misses,
            "errors": metrics.errors
        }


class MemoryOptimizer:
    """Memory optimization and garbage collection."""

    def __init__(self):
        self.gc_threshold = 1000  # Objects before GC
        self.object_count = 0

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        import gc

        # Force garbage collection
        collected = gc.collect()

        # Get memory info
        memory = psutil.virtual_memory()

        return {
            "objects_collected": collected,
            "memory_percent": memory.percent,
            "memory_used": memory.used,
            "memory_available": memory.available
        }

    def monitor_object_creation(self, obj: Any) -> None:
        """Monitor object creation for GC triggers."""
        self.object_count += 1
        if self.object_count >= self.gc_threshold:
            self.optimize_memory_usage()
            self.object_count = 0


class QueryOptimizer:
    """Optimizes queries for code analysis operations."""

    def __init__(self):
        self.query_cache = MultiLevelCache(
            MemoryCache(max_size=500),
            DiskCache(".query_cache")
        )

    def optimize_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a query based on context."""
        cache_key = f"{query}_{hash(str(context))}"

        # Check cache first
        cached_result = self.query_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Perform optimization
        optimized = self._perform_optimization(query, context)

        # Cache result
        self.query_cache.set(cache_key, optimized, ttl=3600)  # 1 hour TTL

        return optimized

    def _perform_optimization(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform actual query optimization."""
        # Analyze query complexity
        complexity = self._analyze_complexity(query)

        # Suggest optimizations
        suggestions = []
        if complexity > 10:
            suggestions.append("Consider breaking down complex query")
        if "SELECT *" in query.upper():
            suggestions.append("Avoid SELECT * for better performance")
        if context.get("large_dataset", False):
            suggestions.append("Consider pagination for large datasets")

        return {
            "original_query": query,
            "complexity_score": complexity,
            "optimizations": suggestions,
            "estimated_improvement": f"{complexity * 10}%"
        }

    def _analyze_complexity(self, query: str) -> int:
        """Analyze query complexity."""
        complexity = 0
        complexity += query.count("JOIN") * 2
        complexity += query.count("WHERE") * 1
        complexity += query.count("GROUP BY") * 1
        complexity += query.count("ORDER BY") * 1
        complexity += len(query.split()) // 10  # Length factor
        return complexity


# Decorators for performance monitoring
def profile_performance(operation_name: str):
    """Decorator to profile function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            operation_id = profiler.start_operation(operation_name)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_operation(operation_id)

        return wrapper
    return decorator


def cache_result(ttl: Optional[int] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        cache = MultiLevelCache(MemoryCache(), DiskCache())

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"

            # Check cache
            cached = cache.get(key)
            if cached is not None:
                return cached

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result, ttl)
            return result

        return wrapper
    return decorator


class AutoScaler:
    """Automatic scaling based on load and performance metrics."""

    def __init__(self, min_workers: int = 1, max_workers: int = 10):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.executor = ThreadPoolExecutor(max_workers=self.current_workers)

    def adjust_workers(self, metrics: Dict[str, Any]) -> None:
        """Adjust worker count based on performance metrics."""
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)
        queue_size = metrics.get("queue_size", 0)

        # Scale up conditions
        if (cpu_usage > 80 or memory_usage > 80 or queue_size > 10) and self.current_workers < self.max_workers:
            self.current_workers += 1
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)

        # Scale down conditions
        elif (cpu_usage < 30 and memory_usage < 50 and queue_size < 2) and self.current_workers > self.min_workers:
            self.current_workers -= 1
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)

    def get_current_workers(self) -> int:
        """Get current number of workers."""
        return self.current_workers


# Main performance optimization orchestrator
class PerformanceOptimizer:
    """Main orchestrator for all performance optimization features."""

    def __init__(self):
        self.cache = MultiLevelCache(MemoryCache(), DiskCache())
        self.profiler = PerformanceProfiler()
        self.task_manager = AsyncTaskManager()
        self.memory_optimizer = MemoryOptimizer()
        self.query_optimizer = QueryOptimizer()
        self.auto_scaler = AutoScaler()

    def optimize_operation(self, operation_name: str, operation_func: Callable,
                          *args, **kwargs) -> Any:
        """Optimize and execute an operation."""
        # Start profiling
        operation_id = self.profiler.start_operation(operation_name)

        try:
            # Check cache first
            cache_key = f"{operation_name}_{hash(str(args) + str(kwargs))}"
            cached_result = self.cache.get(cache_key)

            if cached_result is not None:
                return cached_result

            # Execute operation
            result = operation_func(*args, **kwargs)

            # Cache result
            self.cache.set(cache_key, result, ttl=1800)  # 30 minutes

            return result

        finally:
            # End profiling
            self.profiler.end_operation(operation_id)

    async def optimize_async_operation(self, operation_name: str,
                                     operation_coro: Callable, *args, **kwargs) -> Any:
        """Optimize and execute an async operation."""
        # Submit to task manager
        result = await self.task_manager.submit_task(
            operation_name,
            lambda: operation_coro(*args, **kwargs)
        )
        return result

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "cache_stats": self._get_cache_stats(),
            "performance_metrics": self.profiler.get_metrics(),
            "memory_stats": self.memory_optimizer.optimize_memory_usage(),
            "active_tasks": self.task_manager.get_active_tasks(),
            "current_workers": self.auto_scaler.get_current_workers()
        }

    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # This would need to be implemented based on cache implementation
        return {
            "memory_cache_size": "N/A",  # Would need cache introspection
            "disk_cache_size": "N/A"
        }


# Export main classes
__all__ = [
    "PerformanceMetrics",
    "CacheStrategy",
    "MemoryCache",
    "DiskCache",
    "MultiLevelCache",
    "AsyncTaskManager",
    "PerformanceProfiler",
    "MemoryOptimizer",
    "QueryOptimizer",
    "AutoScaler",
    "PerformanceOptimizer",
    "profile_performance",
    "cache_result"
]
