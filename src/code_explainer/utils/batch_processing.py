"""Optimized batch processing utilities for efficient code validation."""

from typing import List, Tuple, Callable, Any, Iterator
import logging

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Efficient batch processor with minimal allocations."""
    
    __slots__ = ('batch_size',)
    
    def __init__(self, batch_size: int = 32):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of items to process per batch
        """
        self.batch_size = batch_size
    
    def process(self, items: List[Any], processor: Callable[[List[Any]], List[Any]]) -> List[Any]:
        """Process items in batches with minimal memory overhead.
        
        Args:
            items: Items to process
            processor: Function that processes a batch
        
        Returns:
            Processed results
        """
        if not items:
            return []
        
        batch_size = self.batch_size
        results: List[Any] = []
        results_extend = results.extend  # Cache method reference
        
        for i in range(0, len(items), batch_size):
            batch_results = processor(items[i:i + batch_size])
            results_extend(batch_results)
        
        return results
    
    def validate_batch(self, items: List[str], validators: List[Callable[[str], Tuple[bool, str]]]) -> List[Tuple[bool, str]]:
        """Validate batch of items with multiple validators (vectorized).
        
        Args:
            items: Items to validate
            validators: List of validator functions
        
        Returns:
            List of (is_valid, error_message) tuples
        """
        results: List[Tuple[bool, str]] = []
        results_append = results.append  # Cache method reference
        
        for item in items:
            for validator in validators:
                is_valid, error_msg = validator(item)
                if not is_valid:
                    results_append((False, error_msg))
                    break
            else:
                results_append((True, ""))
        
        return results


class ChunkIterator:
    """Memory-efficient iterator for processing large lists in chunks."""
    
    __slots__ = ('items', 'chunk_size', 'index', '_length')
    
    def __init__(self, items: List[Any], chunk_size: int = 100):
        """Initialize chunk iterator.
        
        Args:
            items: Items to iterate
            chunk_size: Size of each chunk
        """
        self.items = items
        self.chunk_size = chunk_size
        self.index = 0
        self._length = len(items)  # Cache length
    
    def __iter__(self) -> Iterator[List[Any]]:
        """Iterate over chunks."""
        return self
    
    def __next__(self) -> List[Any]:
        """Get next chunk."""
        if self.index >= self._length:
            raise StopIteration
        
        chunk = self.items[self.index:self.index + self.chunk_size]
        self.index += self.chunk_size
        return chunk
