"""Early-exit pattern optimizations for reducing computation."""

from typing import Callable, List, Any, Optional, TypeVar

T = TypeVar('T')


class EarlyExit:
    """Helper class for early-exit patterns."""
    
    @staticmethod
    def find_first(items: List[T], predicate: Callable[[T], bool]) -> Optional[T]:
        """Find first item matching predicate (with early exit).
        
        Args:
            items: Items to search
            predicate: Test function
        
        Returns:
            First matching item or None
        """
        for item in items:
            if predicate(item):
                return item
        return None
    
    @staticmethod
    def all_match(items: List[T], predicate: Callable[[T], bool]) -> bool:
        """Check if all items match predicate (short-circuit False).
        
        Args:
            items: Items to check
            predicate: Test function
        
        Returns:
            True if all match, False otherwise
        """
        for item in items:
            if not predicate(item):
                return False
        return True
    
    @staticmethod
    def any_match(items: List[T], predicate: Callable[[T], bool]) -> bool:
        """Check if any item matches predicate (short-circuit True).
        
        Args:
            items: Items to check
            predicate: Test function
        
        Returns:
            True if any match, False otherwise
        """
        for item in items:
            if predicate(item):
                return True
        return False
    
    @staticmethod
    def find_max_by(items: List[T], key_func: Callable[[T], float], 
                   min_threshold: Optional[float] = None) -> Optional[T]:
        """Find maximum item by key function.
        
        Args:
            items: Items to search
            key_func: Key function
            min_threshold: Optional minimum threshold
        
        Returns:
            Item with maximum key value
        """
        if not items:
            return None
        
        max_item = items[0]
        max_val = key_func(max_item)
        
        if min_threshold is not None and max_val < min_threshold:
            return None
        
        for item in items[1:]:
            val = key_func(item)
            if val > max_val:
                max_val = val
                max_item = item
            
            # Early exit if we find perfect value
            if val >= 1.0:
                break
        
        if min_threshold is not None and max_val < min_threshold:
            return None
        
        return max_item
    
    @staticmethod
    def find_min_by(items: List[T], key_func: Callable[[T], float],
                   max_threshold: Optional[float] = None) -> Optional[T]:
        """Find minimum item by key function.
        
        Args:
            items: Items to search
            key_func: Key function
            max_threshold: Optional maximum threshold
        
        Returns:
            Item with minimum key value
        """
        if not items:
            return None
        
        min_item = items[0]
        min_val = key_func(min_item)
        
        if max_threshold is not None and min_val > max_threshold:
            return None
        
        for item in items[1:]:
            val = key_func(item)
            if val < min_val:
                min_val = val
                min_item = item
            
            # Early exit if we find zero value
            if val <= 0.0:
                break
        
        if max_threshold is not None and min_val > max_threshold:
            return None
        
        return min_item


# Export commonly used functions at module level
find_first = EarlyExit.find_first
all_match = EarlyExit.all_match
any_match = EarlyExit.any_match
find_max_by = EarlyExit.find_max_by
find_min_by = EarlyExit.find_min_by
