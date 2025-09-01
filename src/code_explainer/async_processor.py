"""Async processing utilities for high-performance code explanation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class AsyncCodeExplainer:
    """Async wrapper for code explanation operations."""
    
    def __init__(self, base_explainer, max_workers: int = 4):
        """Initialize async explainer.
        
        Args:
            base_explainer: The base CodeExplainer instance
            max_workers: Maximum number of worker threads
        """
        self.base_explainer = base_explainer
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def explain_async(self, code: str, strategy: str = "enhanced_rag") -> str:
        """Explain code asynchronously.
        
        Args:
            code: Code to explain
            strategy: Explanation strategy
            
        Returns:
            Code explanation
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.base_explainer.explain,
            code,
            strategy
        )
    
    async def batch_explain(
        self,
        codes: List[str],
        strategy: str = "enhanced_rag",
        batch_size: int = 10
    ) -> List[str]:
        """Explain multiple code snippets in batches.
        
        Args:
            codes: List of code snippets to explain
            strategy: Explanation strategy
            batch_size: Number of codes to process concurrently
            
        Returns:
            List of explanations
        """
        explanations = []
        
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            batch_tasks = [
                self.explain_async(code, strategy) for code in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to explain code {i + j}: {result}")
                    explanations.append(f"Error: {result}")
                else:
                    explanations.append(result)
        
        return explanations
    
    async def stream_explanations(
        self,
        codes: List[str],
        strategy: str = "enhanced_rag",
        callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> None:
        """Stream explanations as they complete.
        
        Args:
            codes: List of code snippets to explain
            strategy: Explanation strategy
            callback: Optional callback for each completed explanation
        """
        tasks = [
            self.explain_async(code, strategy) for code in codes
        ]
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                explanation = await task
                if callback:
                    await callback(i, explanation)
                else:
                    print(f"Explanation {i + 1}/{len(codes)}: {explanation[:100]}...")
            except Exception as e:
                logger.error(f"Failed to explain code {i}: {e}")
                if callback:
                    await callback(i, f"Error: {e}")
    
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
