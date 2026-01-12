"""Language detection utilities with fast substring heuristics.

Optimized for performance with:
- Early exit on first pattern match
- Pre-compiled pattern tuples (faster iteration than frozenset)
- Hash-based code fingerprinting for cache efficiency
"""

from functools import lru_cache
from hashlib import md5

# Pre-compiled patterns as tuples for faster iteration (vs frozenset)
# Ordered by likelihood/frequency for early exit optimization
_CPP_PATTERNS = ('#include', 'std::', '::')
_JAVA_PATTERNS = ('public static void main', 'system.out', 'public class')
_JS_PATTERNS = ('function ', '=>', 'console.log', 'const ', 'let ')
_PYTHON_PATTERNS = ('def ', 'import ', 'class ', 'print(', 'self.')


def _compute_code_hash(code: str) -> str:
    """Compute hash for code fingerprinting (faster cache key)."""
    return md5(code.encode('utf-8', errors='ignore')).hexdigest()[:16]


@lru_cache(maxsize=4096)
def _detect_language_cached(code_hash: str, code_prefix: str) -> str:
    """Language detector with hash-based caching.
    
    Uses code hash + prefix for cache key to handle large code snippets
    while maintaining cache efficiency.
    
    Returns one of: python, javascript, java, cpp.
    """
    code_l = code_prefix.lower()
    
    # Early exit pattern matching - check most distinctive patterns first
    # C++ detection (most distinctive markers)
    for p in _CPP_PATTERNS:
        if p in code_l:
            return "cpp"
    
    # Java detection
    for p in _JAVA_PATTERNS:
        if p in code_l:
            return "java"
    
    # JavaScript detection
    for p in _JS_PATTERNS:
        if p in code_l:
            return "javascript"
    
    # Python is default but check for confirmation
    for p in _PYTHON_PATTERNS:
        if p in code_l:
            return "python"
    
    return "python"


def detect_language(code: str) -> str:
    """Public API with optimized caching strategy.
    
    Uses hash-based fingerprinting for efficient caching of large code.
    Prefix-based detection reduces memory usage while maintaining accuracy.
    """
    if not code:
        return "python"
    
    # Use first 2000 chars for detection (sufficient for language hints)
    code_prefix = code[:2000] if len(code) > 2000 else code
    code_hash = _compute_code_hash(code_prefix)
    
    return _detect_language_cached(code_hash, code_prefix)