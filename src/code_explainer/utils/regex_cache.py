"""Regex compilation optimization utilities."""

import re
from typing import Dict, Pattern, Optional
from functools import lru_cache


class RegexCache:
    """Cache for pre-compiled regex patterns."""
    
    __slots__ = ('max_patterns', 'cache', '_hits', '_misses')
    
    def __init__(self, max_patterns: int = 100):
        """Initialize regex cache.
        
        Args:
            max_patterns: Maximum number of patterns to cache
        """
        self.max_patterns = max_patterns
        self.cache: Dict[str, Pattern[str]] = {}
        self._hits = 0
        self._misses = 0
    
    def compile(self, pattern: str, flags: int = 0) -> Pattern[str]:
        """Get compiled pattern from cache or compile new.
        
        Args:
            pattern: Regex pattern string
            flags: Regex flags (re.IGNORECASE, etc.)
        
        Returns:
            Compiled regex pattern
        """
        cache_key = f"{pattern}:{flags}"
        
        compiled = self.cache.get(cache_key)
        if compiled is not None:
            self._hits += 1
            return compiled
        
        self._misses += 1
        # Compile new pattern
        compiled = re.compile(pattern, flags)
        
        # Only cache if not full
        if len(self.cache) < self.max_patterns:
            self.cache[cache_key] = compiled
        
        return compiled
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def match(self, pattern: str, text: str, flags: int = 0) -> bool:
        """Check if pattern matches text.
        
        Args:
            pattern: Regex pattern string
            text: Text to match
            flags: Regex flags
        
        Returns:
            True if pattern matches
        """
        return self.compile(pattern, flags).match(text) is not None
    
    def search(self, pattern: str, text: str, flags: int = 0) -> Optional[re.Match[str]]:
        """Search for pattern in text.
        
        Args:
            pattern: Regex pattern string
            text: Text to search
            flags: Regex flags
        
        Returns:
            Match object or None
        """
        return self.compile(pattern, flags).search(text)
    
    def findall(self, pattern: str, text: str, flags: int = 0):
        """Find all matches of pattern in text.
        
        Args:
            pattern: Regex pattern string
            text: Text to search
            flags: Regex flags
        
        Returns:
            List of matches
        """
        return self.compile(pattern, flags).findall(text)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0


# Global regex cache instance
_global_cache: RegexCache = RegexCache()


def get_regex(pattern: str, flags: int = 0) -> Pattern:
    """Get cached compiled regex pattern."""
    return _global_cache.compile(pattern, flags)


def regex_match(pattern: str, text: str, flags: int = 0) -> bool:
    """Quick regex match check."""
    return _global_cache.match(pattern, text, flags)


def regex_search(pattern: str, text: str, flags: int = 0):
    """Quick regex search."""
    return _global_cache.search(pattern, text, flags)


def regex_findall(pattern: str, text: str, flags: int = 0):
    """Quick regex find all."""
    return _global_cache.findall(pattern, text, flags)
