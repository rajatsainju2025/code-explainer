"""Async batch task processor for efficient concurrent task execution.

This module provides optimized batch processing for async operations
to reduce event loop overhead and improve concurrency.

Optimized with:
- Task result pooling to reduce object allocations
- Pre-allocated deque for task queueing
- Lock-free stats tracking where possible
"""

import asyncio
from typing import Any, Callable, List, Optional, Dict, Coroutine
from dataclasses import dataclass
from collections import deque
import threading
import time

# Pool of reusable TaskResult objects
_RESULT_POOL: deque = deque(maxlen=200)
_RESULT_POOL_LOCK = threading.Lock()


def _acquire_result(task_id: str, success: bool, result: Any = None, 
                   error: Optional[Exception] = None, execution_time: float = 0.0) -> 'TaskResult':
    """Acquire TaskResult from pool or create new."""
    with _RESULT_POOL_LOCK:
        if _RESULT_POOL:
            task_result = _RESULT_POOL.popleft()
            task_result.task_id = task_id
            task_result.success = success
            task_result.result = result
            task_result.error = error
            task_result.execution_time = execution_time
            return task_result
    return TaskResult(task_id, success, result, error, execution_time)


def _release_result(task_result: 'TaskResult') -> None:
    """Return TaskResult to pool."""
    with _RESULT_POOL_LOCK:
        if len(_RESULT_POOL) < 200:
            task_result.result = None
            task_result.error = None
            _RESULT_POOL.append(task_result)


@dataclass
class TaskResult:
    """Result of a batch task execution (mutable for pooling)."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    
    def reset(self):
        """Reset for reuse."""
        self.result = None
        self.error = None
        self.execution_time = 0.0


class BatchTaskExecutor:
    """Executor for batch async tasks with intelligent batching."""
    
    __slots__ = ('_batch_size', '_timeout', '_queue', '_lock', '_stats')
    
    def __init__(self, batch_size: int = 16, timeout: float = 30.0):
        """Initialize batch task executor.
        
        Args:
            batch_size: Maximum tasks per batch
            timeout: Timeout per batch in seconds
        """
        self._batch_size = batch_size
        self._timeout = timeout
        self._queue: deque = deque()
        self._lock = threading.RLock()
        self._stats = {
            'total_batches': 0,
            'total_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0.0,
        }
    
    async def execute_batch(self, tasks: List[Callable]) -> List[TaskResult]:
        """Execute multiple tasks as a batch.
        
        Args:
            tasks: List of async callables to execute
            
        Returns:
            List of TaskResult objects
        """
        if not tasks:
            return []
        
        # Break into smaller batches if needed
        results: List[TaskResult] = []
        for i in range(0, len(tasks), self._batch_size):
            batch = tasks[i:i + self._batch_size]
            batch_results = await self._execute_batch_chunk(batch)
            results.extend(batch_results)
        
        return results
    
    async def _execute_batch_chunk(self, tasks: List[Callable]) -> List[TaskResult]:
        """Execute a single batch chunk."""
        start_time = time.time()
        
        # Create coroutines
        coros = [self._wrap_task(task, i) for i, task in enumerate(tasks)]
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*coros, return_exceptions=True),
                timeout=self._timeout
            )
        except asyncio.TimeoutError:
            results = [
                _acquire_result(
                    task_id=f"task_{i}",
                    success=False,
                    error=asyncio.TimeoutError("Batch execution timeout")
                )
                for i in range(len(tasks))
            ]
        
        # Update stats
        with self._lock:
            self._stats['total_batches'] += 1
            self._stats['total_tasks'] += len(tasks)
            self._stats['total_time'] += time.time() - start_time
            failed_count = sum(1 for r in results if isinstance(r, TaskResult) and not r.success)
            self._stats['failed_tasks'] += failed_count
        
        return results
    
    async def _wrap_task(self, task: Callable, task_id: int) -> TaskResult:
        """Wrap task execution with error handling."""
        start = time.time()
        try:
            result = await task() if asyncio.iscoroutinefunction(task) else task()
            return TaskResult(
                task_id=f"task_{task_id}",
                success=True,
                result=result,
                execution_time=time.time() - start
            )
        except Exception as e:
            return TaskResult(
                task_id=f"task_{task_id}",
                success=False,
                error=e,
                execution_time=time.time() - start
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self._lock:
            stats = self._stats.copy()
            if stats['total_batches'] > 0:
                stats['avg_batch_time'] = stats['total_time'] / stats['total_batches']
                stats['success_rate'] = (
                    (stats['total_tasks'] - stats['failed_tasks']) / stats['total_tasks']
                )
            return stats


class ConcurrentTaskLimiter:
    """Limits concurrent execution of tasks to prevent resource exhaustion."""
    
    __slots__ = ('_semaphore', '_pending', '_lock', '_max_concurrent')
    
    def __init__(self, max_concurrent: int = 100):
        """Initialize task limiter.
        
        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending: List[asyncio.Task] = []
        self._lock = threading.RLock()
        self._max_concurrent = max_concurrent
    
    async def run_limited(self, coro: Coroutine) -> Any:
        """Run coroutine with concurrency limit.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Coroutine result
        """
        async with self._semaphore:
            return await coro
    
    async def run_many_limited(self, coros: List[Coroutine]) -> List[Any]:
        """Run multiple coroutines with concurrency limit.
        
        Args:
            coros: List of coroutines
            
        Returns:
            List of results
        """
        tasks = [self.run_limited(coro) for coro in coros]
        return await asyncio.gather(*tasks)
    
    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        with self._lock:
            return len([t for t in self._pending if not t.done()])


class TaskQueue:
    """Queue for managing async tasks with priority support."""
    
    __slots__ = ('_queue', '_lock', '_priority_levels')
    
    def __init__(self, priority_levels: int = 3):
        """Initialize task queue.
        
        Args:
            priority_levels: Number of priority levels
        """
        self._queue: Dict[int, deque] = {i: deque() for i in range(priority_levels)}
        self._lock = threading.RLock()
        self._priority_levels = priority_levels
    
    def enqueue(self, task: Callable, priority: int = 0) -> None:
        """Enqueue a task.
        
        Args:
            task: Task callable
            priority: Priority level (0=lowest, priority_levels-1=highest)
        """
        priority = max(0, min(priority, self._priority_levels - 1))
        with self._lock:
            self._queue[priority].append(task)
    
    def dequeue_batch(self, batch_size: int) -> List[Callable]:
        """Dequeue a batch of tasks, respecting priority.
        
        Args:
            batch_size: Maximum batch size
            
        Returns:
            List of tasks
        """
        with self._lock:
            batch = []
            # Process from highest to lowest priority
            for priority in range(self._priority_levels - 1, -1, -1):
                queue = self._queue[priority]
                while queue and len(batch) < batch_size:
                    batch.append(queue.popleft())
            
            return batch
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return all(len(q) == 0 for q in self._queue.values())
    
    def get_size(self) -> int:
        """Get total queue size."""
        with self._lock:
            return sum(len(q) for q in self._queue.values())
    
    def get_size_by_priority(self) -> Dict[int, int]:
        """Get queue size for each priority level."""
        with self._lock:
            return {p: len(q) for p, q in self._queue.items()}


class AsyncBatchProcessor:
    """High-level async batch processor with queue and priority support."""
    
    __slots__ = ('_executor', '_limiter', '_queue', '_worker_task')
    
    def __init__(self, batch_size: int = 16, max_concurrent: int = 100,
                 priority_levels: int = 3):
        """Initialize async batch processor.
        
        Args:
            batch_size: Tasks per batch
            max_concurrent: Maximum concurrent tasks
            priority_levels: Number of priority levels
        """
        self._executor = BatchTaskExecutor(batch_size=batch_size)
        self._limiter = ConcurrentTaskLimiter(max_concurrent=max_concurrent)
        self._queue = TaskQueue(priority_levels=priority_levels)
        self._worker_task: Optional[asyncio.Task] = None
    
    async def submit_task(self, task: Callable, priority: int = 0) -> None:
        """Submit task to processor.
        
        Args:
            task: Async callable to execute
            priority: Priority level
        """
        self._queue.enqueue(task, priority=priority)
    
    async def process_queue(self, batch_size: Optional[int] = None) -> List[TaskResult]:
        """Process all queued tasks.
        
        Args:
            batch_size: Optional override for batch size
            
        Returns:
            List of all task results
        """
        all_results: List[TaskResult] = []
        
        while not self._queue.is_empty():
            batch = self._queue.dequeue_batch(batch_size or 16)
            if batch:
                batch_results = await self._executor.execute_batch(batch)
                all_results.extend(batch_results)
        
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'executor': self._executor.get_stats(),
            'queue_size': self._queue.get_size(),
            'pending_tasks': self._limiter.get_pending_count(),
        }
