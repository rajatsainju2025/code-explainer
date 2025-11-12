"""String interning utilities for memory and comparison efficiency.

This module provides string interning and common constant pre-interning
to reduce memory footprint and speed up string comparisons.
"""

import sys
import threading
from typing import Dict, Set, List, Optional


class StringInterningPool:
    """Pool for interned strings to reduce memory duplication."""
    
    __slots__ = ('_interned', '_lock', '_stats')
    
    def __init__(self):
        """Initialize string interning pool."""
        self._interned: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._stats = {'total_strings': 0, 'unique_strings': 0, 'memory_saved': 0}
    
    def intern(self, string: str) -> str:
        """Intern a string.
        
        Args:
            string: String to intern
            
        Returns:
            Interned string (reused if already interned)
        """
        with self._lock:
            if string not in self._interned:
                self._interned[string] = string
                self._stats['unique_strings'] += 1
            
            self._stats['total_strings'] += 1
            return self._interned[string]
    
    def intern_many(self, strings: List[str]) -> List[str]:
        """Intern multiple strings.
        
        Args:
            strings: List of strings to intern
            
        Returns:
            List of interned strings
        """
        return [self.intern(s) for s in strings]
    
    def get_stats(self) -> Dict[str, int]:
        """Get interning statistics."""
        with self._lock:
            return self._stats.copy()
    
    def clear(self) -> None:
        """Clear interned strings."""
        with self._lock:
            self._interned.clear()
            self._stats = {'total_strings': 0, 'unique_strings': 0, 'memory_saved': 0}


class ConstantStrings:
    """Pre-interned constant strings for fast access."""
    
    __slots__ = ()
    
    # Common statuses
    SUCCESS = sys.intern("success")
    FAILURE = sys.intern("failure")
    PENDING = sys.intern("pending")
    PROCESSING = sys.intern("processing")
    
    # Common strategies
    VANILLA = sys.intern("vanilla")
    DETAILED = sys.intern("detailed")
    CONCISE = sys.intern("concise")
    
    # Common fields
    CODE = sys.intern("code")
    STRATEGY = sys.intern("strategy")
    EXPLANATION = sys.intern("explanation")
    MODEL_NAME = sys.intern("model_name")
    DEVICE = sys.intern("device")
    DTYPE = sys.intern("dtype")
    MAX_LENGTH = sys.intern("max_length")
    
    # HTTP
    GET = sys.intern("GET")
    POST = sys.intern("POST")
    PUT = sys.intern("PUT")
    DELETE = sys.intern("DELETE")
    PATCH = sys.intern("PATCH")
    
    # Errors
    ERROR = sys.intern("error")
    ERROR_MESSAGE = sys.intern("error_message")
    DETAIL = sys.intern("detail")
    REQUEST_ID = sys.intern("request_id")
    
    # Response fields
    RESULTS = sys.intern("results")
    COUNT = sys.intern("count")
    TOTAL = sys.intern("total")
    PAGE = sys.intern("page")
    PROCESSING_TIME = sys.intern("processing_time")


class FastStringComparison:
    """Utilities for fast string comparison using interning."""
    
    __slots__ = ('_pool',)
    
    def __init__(self, pool: Optional[StringInterningPool] = None):
        """Initialize fast comparison.
        
        Args:
            pool: String interning pool to use
        """
        self._pool = pool or StringInterningPool()
    
    def equals_cached(self, s1: str, s2: str) -> bool:
        """Check if two strings are equal using cached lookup.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            True if strings are equal
        """
        # Intern both strings for identity comparison
        interned1 = self._pool.intern(s1)
        interned2 = self._pool.intern(s2)
        # Identity comparison is O(1)
        return interned1 is interned2
    
    def in_set_cached(self, string: str, string_set: Set[str]) -> bool:
        """Check if string is in set using interning.
        
        Args:
            string: String to check
            string_set: Set of strings
            
        Returns:
            True if string is in set
        """
        interned = self._pool.intern(string)
        # Set lookup with interned strings
        for s in string_set:
            if interned is self._pool.intern(s):
                return True
        return False


class CachedStringLookup:
    """Cached lookups for strings with compile-time optimization."""
    
    __slots__ = ('_cache', '_lock')
    
    def __init__(self):
        """Initialize cached lookup."""
        self._cache: Dict[str, bool] = {}
        self._lock = threading.RLock()
    
    def is_valid_identifier(self, string: str) -> bool:
        """Check if string is valid Python identifier using cache.
        
        Args:
            string: String to check
            
        Returns:
            True if valid identifier
        """
        with self._lock:
            if string not in self._cache:
                self._cache[string] = string.isidentifier()
            return self._cache[string]
    
    def starts_with_underscore(self, string: str) -> bool:
        """Check if string starts with underscore."""
        return string and string[0] == '_'
    
    def is_numeric(self, string: str) -> bool:
        """Check if string is numeric."""
        try:
            float(string)
            return True
        except (ValueError, TypeError):
            return False


# Global singleton pool
_global_pool: Optional[StringInterningPool] = None
_pool_lock = threading.RLock()


def get_string_pool() -> StringInterningPool:
    """Get global string interning pool."""
    global _global_pool
    
    if _global_pool is None:
        with _pool_lock:
            if _global_pool is None:
                _global_pool = StringInterningPool()
                # Pre-intern common strings
                for attr_name in dir(ConstantStrings):
                    if not attr_name.startswith('_'):
                        value = getattr(ConstantStrings, attr_name)
                        if isinstance(value, str):
                            _global_pool.intern(value)
    
    return _global_pool


def intern_string(string: str) -> str:
    """Intern a string using the global pool.
    
    Args:
        string: String to intern
        
    Returns:
        Interned string
    """
    return get_string_pool().intern(string)


def intern_many_strings(strings: List[str]) -> List[str]:
    """Intern multiple strings.
    
    Args:
        strings: Strings to intern
        
    Returns:
        Interned strings
    """
    return get_string_pool().intern_many(strings)
