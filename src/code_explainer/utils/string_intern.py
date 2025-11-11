"""String interning utilities for efficient keyword matching and comparison."""

import sys
from typing import FrozenSet, Set


class StringIntern:
    """String pool for efficient keyword and identifier matching."""
    
    def __init__(self):
        """Initialize string pool."""
        self._pool: Set[str] = set()
    
    def intern(self, s: str) -> str:
        """Intern a string for comparison efficiency.
        
        Uses Python's sys.intern for common strings.
        Subsequent comparisons use identity (is) instead of equality (==).
        
        Args:
            s: String to intern
        
        Returns:
            Interned string
        """
        # Let Python handle the interning for common strings
        return sys.intern(s)
    
    def intern_set(self, strings: Set[str]) -> FrozenSet[str]:
        """Intern a set of strings and return frozenset.
        
        Args:
            strings: Strings to intern
        
        Returns:
            Frozenset of interned strings
        """
        return frozenset(self.intern(s) for s in strings)


# Pre-intern common strategy keywords
_INTERNED_STRATEGIES = frozenset({
    sys.intern("vanilla"),
    sys.intern("ast_augmented"),
    sys.intern("retrieval_augmented"),
    sys.intern("execution_trace"),
    sys.intern("enhanced_rag"),
})

# Pre-intern common cache strategies
_INTERNED_CACHE_STRATEGIES = frozenset({
    sys.intern("lru"),
    sys.intern("lfu"),
    sys.intern("fifo"),
    sys.intern("size_based"),
    sys.intern("adaptive"),
})

# Pre-intern common device types
_INTERNED_DEVICES = frozenset({
    sys.intern("cpu"),
    sys.intern("cuda"),
    sys.intern("mps"),
    sys.intern("auto"),
})


def get_interned_strategy(strategy: str) -> str:
    """Get interned strategy string for efficient comparison."""
    return sys.intern(strategy)


def is_valid_strategy(strategy: str) -> bool:
    """Check if strategy is valid using interned strings."""
    interned = sys.intern(strategy)
    return interned in _INTERNED_STRATEGIES


def get_interned_cache_strategy(strategy: str) -> str:
    """Get interned cache strategy string."""
    return sys.intern(strategy)


def is_valid_cache_strategy(strategy: str) -> bool:
    """Check if cache strategy is valid."""
    interned = sys.intern(strategy)
    return interned in _INTERNED_CACHE_STRATEGIES


def get_interned_device(device: str) -> str:
    """Get interned device type string."""
    return sys.intern(device)


def is_valid_device(device: str) -> bool:
    """Check if device type is valid."""
    interned = sys.intern(device)
    return interned in _INTERNED_DEVICES
