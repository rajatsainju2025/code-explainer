"""Fast-path utilities for common validation checks."""

from typing import Any, Optional, List, Dict


class FastValidation:
    """Fast-path validation with minimal overhead."""
    
    @staticmethod
    def is_empty_or_none(value: Any) -> bool:
        """Fast check for empty or None values.
        
        Uses short-circuit evaluation for quick rejection.
        """
        return value is None or (hasattr(value, '__len__') and len(value) == 0)
    
    @staticmethod
    def is_valid_string(value: Any, min_length: int = 1) -> bool:
        """Fast validation for string values.
        
        Checks type and length in optimal order.
        """
        if not isinstance(value, str):
            return False
        
        # Early exit for obviously too short
        if len(value) < min_length:
            return False
        
        return True
    
    @staticmethod
    def is_valid_list(value: Any, min_size: int = 0, max_size: Optional[int] = None) -> bool:
        """Fast validation for list values."""
        if not isinstance(value, list):
            return False
        
        length = len(value)
        if length < min_size:
            return False
        
        if max_size is not None and length > max_size:
            return False
        
        return True
    
    @staticmethod
    def first_non_empty(items: List[Optional[str]]) -> Optional[str]:
        """Get first non-empty string from list (fast iteration)."""
        for item in items:
            if item and len(item) > 0:
                return item
        return None
    
    @staticmethod
    def coalesce(*values) -> Any:
        """Return first non-None value (null coalescing)."""
        for value in values:
            if value is not None:
                return value
        return None
    
    @staticmethod
    def safe_get(d: Dict, key: Any, default: Any = None) -> Any:
        """Safe dict access with default.
        
        Optimized for common case where key exists.
        """
        try:
            return d[key]
        except (KeyError, TypeError):
            return default
    
    @staticmethod
    def safe_call(func, *args, default: Any = None, **kwargs) -> Any:
        """Safe function call with exception handling.
        
        Returns default if call fails.
        """
        try:
            return func(*args, **kwargs)
        except Exception:
            return default


# Pre-built validator for common patterns
_validation = FastValidation()

# Export commonly used methods at module level for fast access
is_empty_or_none = _validation.is_empty_or_none
is_valid_string = _validation.is_valid_string
is_valid_list = _validation.is_valid_list
first_non_empty = _validation.first_non_empty
coalesce = _validation.coalesce
safe_get = _validation.safe_get
safe_call = _validation.safe_call
