"""Concurrent processing optimizations for multi-threaded and async operations.

This module provides utilities for efficient concurrent execution with
reduced overhead from thread spawning and synchronization.
"""

import threading
import asyncio
from typing import Any, Callable, List, Optional, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


class OptimizedThreadPool:
    """Optimized thread pool with reuse and minimal overhead."""
    
    __slots__ = ('_executor', '_max_workers', '_thread_count')
    
    def __init__(self, max_workers: int = 10):
        """Initialize thread pool."""
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="opt")
        self._max_workers = max_workers
        self._thread_count = 0
    
    def submit(self, fn: Callable, *args, **kwargs) -> Any:
        """Submit work to thread pool."""
        return self._executor.submit(fn, *args, **kwargs)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread pool."""
        self._executor.shutdown(wait=wait)


class FastAsyncPool:
    """Fast async task pool with queue management."""
    
    __slots__ = ('_loop', '_executor', '_tasks', '_lock')
    
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize async pool."""
        self._loop = loop or asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._tasks: List[asyncio.Task] = []
        self._lock = threading.RLock()
    
    async def run_in_thread(self, fn: Callable, *args, **kwargs) -> Any:
        """Run function in thread pool from async context."""
        return await self._loop.run_in_executor(self._executor, fn, *args)
    
    async def gather(self, *coros: Coroutine) -> List[Any]:
        """Gather coroutine results."""
        return await asyncio.gather(*coros)


class WorkerGroup:
    """Manages group of worker threads."""
    
    __slots__ = ('_workers', '_queue', '_running', '_lock')
    
    def __init__(self, num_workers: int = 4):
        """Initialize worker group."""
        from queue import Queue
        self._workers: List[threading.Thread] = []
        self._queue = Queue()
        self._running = True
        self._lock = threading.RLock()
        
        # Start workers
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self) -> None:
        """Main worker loop."""
        while self._running:
            try:
                fn, args, kwargs = self._queue.get(timeout=1)
                if fn is None:  # Stop signal
                    break
                fn(*args, **kwargs)
            except Exception:
                pass
    
    def submit(self, fn: Callable, *args, **kwargs) -> None:
        """Submit work to worker group."""
        self._queue.put((fn, args, kwargs))
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker group."""
        self._running = False
        if wait:
            for worker in self._workers:
                worker.join(timeout=5)


class ConcurrentCounter:
    """Thread-safe counter with minimal overhead."""
    
    __slots__ = ('_value', '_lock')
    
    def __init__(self):
        """Initialize counter."""
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        """Increment counter."""
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        """Decrement counter."""
        with self._lock:
            self._value -= 1
            return self._value
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value


class DeferredExecutor:
    """Executor for deferred/delayed execution."""
    
    __slots__ = ('_executor', '_timers', '_lock')
    
    def __init__(self):
        """Initialize deferred executor."""
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._timers: List[threading.Timer] = []
        self._lock = threading.RLock()
    
    def schedule_at(self, delay: float, fn: Callable, *args, **kwargs) -> threading.Timer:
        """Schedule function to run after delay."""
        timer = threading.Timer(delay, fn, args=args, kwargs=kwargs)
        timer.start()
        
        with self._lock:
            self._timers.append(timer)
        
        return timer
    
    def schedule_periodic(self, interval: float, fn: Callable, *args, **kwargs) -> 'PeriodicTimer':
        """Schedule periodic execution."""
        return PeriodicTimer(interval, fn, args=args, kwargs=kwargs)


class PeriodicTimer:
    """Periodic timer for recurring tasks."""
    
    __slots__ = ('_interval', '_fn', '_args', '_kwargs', '_running', '_thread')
    
    def __init__(self, interval: float, fn: Callable, args=(), kwargs=None):
        """Initialize periodic timer."""
        self._interval = interval
        self._fn = fn
        self._args = args
        self._kwargs = kwargs or {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start periodic execution."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self) -> None:
        """Run periodic task."""
        while self._running:
            time.sleep(self._interval)
            if self._running:
                try:
                    self._fn(*self._args, **self._kwargs)
                except Exception:
                    pass
    
    def stop(self) -> None:
        """Stop periodic execution."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


# Global instances
_thread_pool = OptimizedThreadPool(max_workers=10)


def get_thread_pool() -> OptimizedThreadPool:
    """Get global thread pool."""
    return _thread_pool
