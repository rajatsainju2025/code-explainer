"""Async processing utilities for high-performance code explanation."""

import asyncio
import logging
import time
import threading
import json
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Rate limiter for controlling request frequency."""
    requests_per_second: float
    burst_limit: int = 10
    _tokens: float = field(init=False, default=0)
    _last_update: float = field(init=False, default_factory=time.time)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self):
        self._tokens = self.burst_limit

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        with self._lock:
            now = time.time()
            time_passed = now - self._last_update
            self._tokens = min(
                self.burst_limit,
                self._tokens + time_passed * self.requests_per_second
            )
            self._last_update = now

            if self._tokens >= 1:
                self._tokens -= 1
            else:
                # Calculate wait time
                wait_time = (1 - self._tokens) / self.requests_per_second
                self._tokens = 0
                await asyncio.sleep(wait_time)


@dataclass
class ConnectionPool:
    """Connection pool for managing resources."""
    max_connections: int
    _available: asyncio.Queue = field(init=False)
    _in_use: set = field(init=False, default_factory=set)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self):
        self._available = asyncio.Queue(maxsize=self.max_connections)
        for i in range(self.max_connections):
            self._available.put_nowait(f"connection_{i}")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        connection = await self._available.get()
        async with self._lock:
            self._in_use.add(connection)

        try:
            yield connection
        finally:
            async with self._lock:
                self._in_use.remove(connection)
            await self._available.put(connection)

    async def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        async with self._lock:
            return {
                "available": self._available.qsize(),
                "in_use": len(self._in_use),
                "total": self.max_connections
            }


class AsyncCodeExplainer:
    """Async wrapper for code explanation operations with advanced features."""

    def __init__(
        self,
        base_explainer,
        max_workers: int = 4,
        rate_limit: Optional[float] = None,
        enable_connection_pool: bool = True
    ):
        """Initialize async explainer.

        Args:
            base_explainer: The base CodeExplainer instance
            max_workers: Maximum number of worker threads
            rate_limit: Requests per second limit
            enable_connection_pool: Whether to use connection pooling
        """
        self.base_explainer = base_explainer
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.rate_limiter = RateLimiter(rate_limit) if rate_limit else None
        self.connection_pool = ConnectionPool(max_workers) if enable_connection_pool else None
        self._semaphore = asyncio.Semaphore(max_workers)

    async def explain_async(
        self,
        code: str,
        strategy: str = "enhanced_rag",
        **kwargs
    ) -> str:
        """Explain code asynchronously with rate limiting and resource management.

        Args:
            code: Code to explain
            strategy: Explanation strategy
            **kwargs: Additional arguments

        Returns:
            Code explanation
        """
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        async with self._semaphore:
            if self.connection_pool:
                async with self.connection_pool.get_connection():
                    return await self._execute_explanation(code, strategy, **kwargs)
            else:
                return await self._execute_explanation(code, strategy, **kwargs)

    async def _execute_explanation(
        self,
        code: str,
        strategy: str,
        **kwargs
    ) -> str:
        """Execute the explanation in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._explain_with_error_handling,
            code,
            strategy,
            kwargs
        )

    def _explain_with_error_handling(
        self,
        code: str,
        strategy: str,
        kwargs: Dict[str, Any]
    ) -> str:
        """Explain code with error handling."""
        try:
            return self.base_explainer.explain_code(code, strategy=strategy, **kwargs)
        except Exception as e:
            logger.error(f"Error explaining code: {e}")
            return f"Error: {str(e)}"

    async def batch_explain(
        self,
        codes: List[str],
        strategy: str = "enhanced_rag",
        batch_size: int = 10,
        **kwargs
    ) -> List[str]:
        """Explain multiple code snippets in batches with advanced features.

        Args:
            codes: List of code snippets to explain
            strategy: Explanation strategy
            batch_size: Number of codes to process concurrently
            **kwargs: Additional arguments

        Returns:
            List of explanations
        """
        semaphore = asyncio.Semaphore(batch_size)
        results: List[Optional[str]] = [None] * len(codes)

        async def explain_with_semaphore(index: int, code: str):
            async with semaphore:
                explanation = await self.explain_async(code, strategy, **kwargs)
                results[index] = explanation

        # Create tasks for all codes
        tasks = [
            explain_with_semaphore(i, code)
            for i, code in enumerate(codes)
        ]

        # Execute all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

        # Convert to List[str], replacing None with empty strings for failed explanations
        return [result if result is not None else "" for result in results]

    async def stream_explanations(
        self,
        codes: List[str],
        strategy: str = "enhanced_rag",
        callback: Optional[Callable[[int, str], Awaitable[None]]] = None,
        **kwargs
    ) -> None:
        """Stream explanations as they complete with enhanced error handling.

        Args:
            codes: List of code snippets to explain
            strategy: Explanation strategy
            callback: Optional callback for each completed explanation
            **kwargs: Additional arguments
        """
        async def explain_with_callback(index: int, code: str):
            try:
                explanation = await self.explain_async(code, strategy, **kwargs)
                if callback:
                    await callback(index, explanation)
                else:
                    print(f"Explanation {index + 1}/{len(codes)}: {explanation[:100]}...")
            except Exception as e:
                error_msg = f"Error: {e}"
                logger.error(f"Failed to explain code {index}: {e}")
                if callback:
                    await callback(index, error_msg)
                else:
                    print(f"Explanation {index + 1}/{len(codes)}: {error_msg}")

        tasks = [
            explain_with_callback(i, code)
            for i, code in enumerate(codes)
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the async processor."""
        stats = {
            "executor_active_threads": self.executor._threads,
            "semaphore_locks": self._semaphore._value,
        }

        if self.connection_pool:
            pool_stats = await self.connection_pool.get_stats()
            stats.update({
                "connection_pool": pool_stats,
                "rate_limiter_active": self.rate_limiter is not None
            })

        return stats

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class BatchProcessor:
    """Batch processing utilities for code explanation tasks."""

    @staticmethod
    async def process_dataset(
        explainer: AsyncCodeExplainer,
        dataset: List[Dict[str, Any]],
        output_file: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Process an entire dataset of code explanations.

        Args:
            explainer: Async code explainer instance
            dataset: Dataset with 'code' field
            output_file: Optional file to save results
            progress_callback: Optional progress callback

        Returns:
            Dataset with added 'explanation' field
        """
        start_time = time.time()
        results = []

        codes = [item.get('code', '') for item in dataset]
        explanations = await explainer.batch_explain(codes)

        for i, (item, explanation) in enumerate(zip(dataset, explanations)):
            result = item.copy()
            result['explanation'] = explanation
            result['processing_time'] = time.time() - start_time
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(dataset))

        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved {len(results)} explanations to {output_file}")

        total_time = time.time() - start_time
        logger.info(f"Processed {len(dataset)} codes in {total_time:.2f}s "
                   f"({len(dataset) / total_time:.2f} codes/sec)")

        return results

    @staticmethod
    def create_progress_callback() -> Callable[[int, int], None]:
        """Create a simple progress callback."""
        def callback(current: int, total: int):
            percentage = (current / total) * 100
            print(f"Progress: {current}/{total} ({percentage:.1f}%)")
        return callback


async def main():
    """Example usage of async code explainer."""
    from code_explainer import CodeExplainer

    # Initialize explainers
    base_explainer = CodeExplainer()
    async_explainer = AsyncCodeExplainer(base_explainer, max_workers=4)

    # Example codes
    codes = [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "class Stack: def __init__(self): self.items = []",
        "import numpy as np; arr = np.array([1, 2, 3])",
    ]

    # Batch processing
    print("Batch processing...")
    explanations = await async_explainer.batch_explain(codes)
    for i, explanation in enumerate(explanations):
        print(f"Code {i + 1}: {explanation[:100]}...")

    # Streaming processing
    print("\nStreaming processing...")
    await async_explainer.stream_explanations(codes)


if __name__ == "__main__":
    asyncio.run(main())
