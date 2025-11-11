"""Regex compilation optimization utilities."""

import re
from typing import Dict, Pattern
from functools import lru_cache


class RegexCache:
    """Cache for pre-compiled regex patterns."""
    
    def __init__(self, max_patterns: int = 100):
        """Initialize regex cache.
        
        Args:
            max_patterns: Maximum number of patterns to cache
        """
        self.max_patterns = max_patterns
        self.cache: Dict[str, Pattern] = {}
    
    def compile(self, pattern: str, flags: int = 0) -> Pattern:
        """Get compiled pattern from cache or compile new.
        
        Args:
            pattern: Regex pattern string
            flags: Regex flags (re.IGNORECASE, etc.)
        
        Returns:
            Compiled regex pattern
        """
        cache_key = f"{pattern}:{flags}"
        
        if cache_key not in self.cache:
            # Compile new pattern if cache not full
            if len(self.cache) < self.max_patterns:
                self.cache[cache_key] = re.compile(pattern, flags)
            else:
                # Just compile without caching
                return re.compile(pattern, flags)
        
        return self.cache[cache_key]
    
    def match(self, pattern: str, text: str, flags: int = 0) -> bool:
        """Check if pattern matches text.
        
        Args:
            pattern: Regex pattern string
            text: Text to match
            flags: Regex flags
        
        Returns:
            True if pattern matches
        """
        compiled = self.compile(pattern, flags)
        return compiled.match(text) is not None
    
    def search(self, pattern: str, text: str, flags: int = 0):
        """Search for pattern in text.
        
        Args:
            pattern: Regex pattern string
            text: Text to search
            flags: Regex flags
        
        Returns:
            Match object or None
        """
        compiled = self.compile(pattern, flags)
        return compiled.search(text)
    
    def findall(self, pattern: str, text: str, flags: int = 0):
        """Find all matches of pattern in text.
        
        Args:
            pattern: Regex pattern string
            text: Text to search
            flags: Regex flags
        
        Returns:
            List of matches
        """
        compiled = self.compile(pattern, flags)
        return compiled.findall(text)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()


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
