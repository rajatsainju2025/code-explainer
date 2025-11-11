"""Generator-based result streaming for memory-efficient processing."""

from typing import Generator, List, Any, Callable, TypeVar, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')


class ResultStreamer:
    """Provides memory-efficient streaming of results via generators."""
    
    @staticmethod
    def batch_stream(items: List[T], batch_size: int = 32) -> Generator[List[T], None, None]:
        """Stream items in batches without loading all into memory.
        
        Args:
            items: Items to stream
            batch_size: Size of each batch
            
        Yields:
            Batches of items
        """
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    @staticmethod
    def filtered_stream(items: List[T], predicate: Callable[[T], bool]) -> Generator[T, None, None]:
        """Stream items that match a predicate.
        
        Args:
            items: Items to filter
            predicate: Filter function
            
        Yields:
            Filtered items
        """
        for item in items:
            if predicate(item):
                yield item
    
    @staticmethod
    def map_stream(items: List[T], mapper: Callable[[T], U]) -> Generator[U, None, None]:
        """Stream transformed items without intermediate list.
        
        Args:
            items: Items to transform
            mapper: Transformation function
            
        Yields:
            Transformed items
        """
        for item in items:
            yield mapper(item)
    
    @staticmethod
    def deduplicate_stream(items: List[T]) -> Generator[T, None, None]:
        """Stream items with deduplication (maintains order).
        
        Args:
            items: Items to deduplicate
            
        Yields:
            Unique items in original order
        """
        seen = set()
        for item in items:
            # Handle unhashable types gracefully
            try:
                if item not in seen:
                    seen.add(item)
                    yield item
            except TypeError:
                # Unhashable type, always yield
                yield item
    
    @staticmethod
    def chain_streams(*generators: Generator[T, None, None]) -> Generator[T, None, None]:
        """Chain multiple generators into one.
        
        Args:
            *generators: Generators to chain
            
        Yields:
            Items from all generators in order
        """
        for gen in generators:
            yield from gen
    
    @staticmethod
    def take(generator: Generator[T, None, None], n: int) -> Generator[T, None, None]:
        """Take first n items from generator.
        
        Args:
            generator: Generator to take from
            n: Number of items to take
            
        Yields:
            First n items
        """
        for i, item in enumerate(generator):
            if i >= n:
                break
            yield item
    
    @staticmethod
    def skip(generator: Generator[T, None, None], n: int) -> Generator[T, None, None]:
        """Skip first n items from generator.
        
        Args:
            generator: Generator to skip from
            n: Number of items to skip
            
        Yields:
            Items after skipping first n
        """
        for i, item in enumerate(generator):
            if i >= n:
                yield item
    
    @staticmethod
    def consume_stream(generator: Generator[T, None, None]) -> List[T]:
        """Consume all items from generator (use sparingly).
        
        Args:
            generator: Generator to consume
            
        Returns:
            List of all items
        """
        return list(generator)


# Export commonly used methods at module level
batch_stream = ResultStreamer.batch_stream
filtered_stream = ResultStreamer.filtered_stream
map_stream = ResultStreamer.map_stream
deduplicate_stream = ResultStreamer.deduplicate_stream
chain_streams = ResultStreamer.chain_streams
take = ResultStreamer.take
skip = ResultStreamer.skip
consume_stream = ResultStreamer.consume_stream
