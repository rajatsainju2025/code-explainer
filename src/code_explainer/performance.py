"""Performance optimization utilities for code explanation."""

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import threading
import queue
import logging

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Optimize performance of code explanation operations."""

    def __init__(self):
        self.operation_stats = defaultdict(list)
        self.lock = threading.Lock()

    def memoize(self, maxsize: int = 128):
        """Memoization decorator with LRU cache."""
        def decorator(func: Callable) -> Callable:
            cache = {}
            cache_order = []

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = str(args) + str(sorted(kwargs.items()))

                with self.lock:
                    if key in cache:
                        # Move to end (most recently used)
                        cache_order.remove(key)
                        cache_order.append(key)
                        return cache[key]

                    # Compute result
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()

                    # Store in cache
                    cache[key] = result
                    cache_order.append(key)

                    # Remove oldest if cache is full
                    if len(cache) > maxsize:
                        oldest = cache_order.pop(0)
                        del cache[oldest]

                    # Record stats
                    self.operation_stats[func.__name__].append(end_time - start_time)

                    return result

            def cache_info():
                return {
                    'hits': len([k for k in cache_order if k in cache]),
                    'misses': len(self.operation_stats[func.__name__]),
                    'maxsize': maxsize,
                    'currsize': len(cache)
                }

            def cache_clear():
                cache.clear()
                cache_order.clear()

            wrapper.cache_info = cache_info  # type: ignore
            wrapper.cache_clear = cache_clear  # type: ignore

            return wrapper
        return decorator

    def batch_optimize(self, batch_size: int = 10):
        """Batch optimization decorator."""
        def decorator(func: Callable) -> Callable:
            batch_queue = queue.Queue()

            @functools.wraps(func)
            def wrapper(item):
                batch_queue.put(item)

                if batch_queue.qsize() >= batch_size:
                    # Process batch
                    batch = []
                    while not batch_queue.empty() and len(batch) < batch_size:
                        batch.append(batch_queue.get())

                    return func(batch)
                else:
                    # Return placeholder for small batches
                    return func([item])[0]

            return wrapper
        return decorator

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics."""
        report = {}

        for operation, times in self.operation_stats.items():
            if times:
                report[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times),
                    'median_time': np.median(times),
                    'p95_time': np.percentile(times, 95),
                    'p99_time': np.percentile(times, 99)
                }

        return report


class CodePreprocessor:
    """Optimize code preprocessing for better performance."""

    @staticmethod
    def normalize_code(code: str) -> str:
        """Normalize code for better caching and processing.

        Args:
            code: Raw code string

        Returns:
            Normalized code string
        """
        # Remove extra whitespace
        lines = [line.rstrip() for line in code.splitlines()]

        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Normalize indentation
        if lines:
            # Find minimum indentation
            min_indent = min(
                len(line) - len(line.lstrip())
                for line in lines
                if line.strip()
            )

            # Remove common indentation
            normalized_lines = []
            for line in lines:
                if line.strip():
                    normalized_lines.append(line[min_indent:])
                else:
                    normalized_lines.append('')
        else:
            normalized_lines = lines

        return '\n'.join(normalized_lines)

    @staticmethod
    def extract_code_features(code: str) -> Dict[str, Any]:
        """Extract features for similarity matching and caching.

        Args:
            code: Code to analyze

        Returns:
            Dictionary of code features
        """
        import ast
        import hashlib

        features = {
            'length': len(code),
            'lines': len(code.splitlines()),
            'hash': hashlib.md5(code.encode()).hexdigest(),
        }

        try:
            tree = ast.parse(code)

            # Count different node types
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1

            features.update({
                'has_functions': node_counts.get('FunctionDef', 0) > 0,
                'has_classes': node_counts.get('ClassDef', 0) > 0,
                'has_loops': (node_counts.get('For', 0) + node_counts.get('While', 0)) > 0,
                'has_imports': (node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0)) > 0,
                'complexity_score': sum(node_counts.values()),
                'function_count': node_counts.get('FunctionDef', 0),
                'class_count': node_counts.get('ClassDef', 0),
                'node_counts': dict(node_counts)
            })

        except SyntaxError:
            features.update({
                'has_functions': False,
                'has_classes': False,
                'has_loops': False,
                'has_imports': False,
                'complexity_score': 0,
                'function_count': 0,
                'class_count': 0,
                'syntax_error': True
            })

        return features


class BatchCache:
    """Intelligent caching for batch operations."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Update access order
                self.access_order.remove(key)
                self.access_order.append(key)
                self.stats['hits'] += 1
                return self.cache[key]
            else:
                self.stats['misses'] += 1
                return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
                self.access_order.append(key)
                self.cache[key] = value
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Evict oldest
                    oldest = self.access_order.pop(0)
                    del self.cache[oldest]
                    self.stats['evictions'] += 1

                self.cache[key] = value
                self.access_order.append(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                **self.stats
            }

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}


# Global performance optimizer instance
optimizer = PerformanceOptimizer()

# Common decorators
memoize = optimizer.memoize
batch_optimize = optimizer.batch_optimize
