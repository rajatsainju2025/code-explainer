"""Memory profiling utilities for code-explainer."""

import gc
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import sys


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryStats:
    """Memory usage statistics."""
    
    allocated_mb: float
    peak_mb: float
    gc_count: int
    object_count: int


class MemoryProfiler:
    """Profile memory usage during operations."""
    
    __slots__ = ('_initial_stats', '_enabled')

    def __init__(self):
        """Initialize memory profiler."""
        self._initial_stats: Optional[MemoryStats] = None
        self._enabled = True
    
    def __enter__(self):
        """Enter context - capture initial memory state."""
        if self._enabled:
            self._initial_stats = self.get_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - report memory delta and cleanup."""
        if self._enabled and self._initial_stats:
            final_stats = self.get_memory_stats()
            delta_mb = final_stats.allocated_mb - self._initial_stats.allocated_mb
            
            if abs(delta_mb) > 1.0:  # Only log if significant change
                logger.info(
                    f"Memory delta: {delta_mb:+.2f} MB "
                    f"(peak: {final_stats.peak_mb:.2f} MB, "
                    f"gc_count: {final_stats.gc_count})"
                )
            
            # Trigger cleanup
            gc.collect()
        
        return False
    
    @staticmethod
    def get_memory_stats() -> MemoryStats:
        """Get current memory statistics.
        
        Returns:
            MemoryStats object with current memory info
        """
        # Get garbage collector stats
        gc_stats = gc.get_stats()
        gc_count = sum(stat.get('collections', 0) for stat in gc_stats)
        
        # Count tracked objects
        object_count = len(gc.get_objects())
        
        # Try to get process memory if available
        allocated_mb = 0.0
        peak_mb = 0.0
        
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            allocated_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
            peak_mb = allocated_mb  # psutil doesn't track peak on all platforms
        except ImportError:
            # Fallback to sys.getsizeof for rough estimate
            allocated_mb = sys.getsizeof(gc.get_objects()) / (1024 * 1024)
            peak_mb = allocated_mb
        
        return MemoryStats(
            allocated_mb=allocated_mb,
            peak_mb=peak_mb,
            gc_count=gc_count,
            object_count=object_count
        )
    
    @staticmethod
    def force_cleanup() -> Dict[str, int]:
        """Force garbage collection and return collected counts.
        
        Returns:
            Dictionary with collection counts per generation
        """
        counts = {}
        for generation in range(gc.get_count().__len__()):
            collected = gc.collect(generation)
            counts[f"gen_{generation}"] = collected
        
        return counts
