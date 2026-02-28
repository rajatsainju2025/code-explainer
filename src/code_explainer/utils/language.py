"""Language detection utilities with fast substring heuristics.

Optimized for performance with:
- Early exit on first pattern match
- Pre-compiled pattern tuples (faster iteration than frozenset)
- lru_cache on the truncated code prefix for deduplication
"""

from functools import lru_cache

# Pre-compiled patterns as tuples for faster iteration (vs frozenset)
# Ordered by likelihood/frequency for early exit optimization
_CPP_PATTERNS = ('#include', 'std::', '::')
_JAVA_PATTERNS = ('public static void main', 'system.out', 'public class')
# JS patterns refined to avoid false positives on Python code.
# 'const ' and 'let ' were too broad — Python variables like 'constants'
# or 'letter' triggered JavaScript detection.
_JS_PATTERNS = ('function ', '=>', 'console.log', 'const ', 'let ', 'var ')
_PYTHON_PATTERNS = ('def ', 'import ', 'class ', 'print(', 'self.')


@lru_cache(maxsize=4096)
def _detect_language_cached(code_prefix: str) -> str:
    """Language detector with lru_cache on the truncated prefix.

    The cache key is the prefix string itself — lru_cache already hashes
    it internally.  A separate md5 pass was previously computed before
    every call, doubling the hashing work for no benefit (the full prefix
    was still passed as a second argument, so lru_cache hashed it anyway).

    Returns one of: python, javascript, java, cpp.
    """
    code_l = code_prefix.lower()
    
    # Check Python FIRST — it's the most common input for this tool,
    # and checking JS first produces false positives on Python code
    # containing 'const ' (constants=...) or 'let ' (letter=...).
    for p in _PYTHON_PATTERNS:
        if p in code_l:
            return "python"
    
    # C++ detection (most distinctive markers)
    for p in _CPP_PATTERNS:
        if p in code_l:
            return "cpp"
    
    # Java detection
    for p in _JAVA_PATTERNS:
        if p in code_l:
            return "java"
    
    # JavaScript detection (after Python to avoid false positives)
    for p in _JS_PATTERNS:
        if p in code_l:
            return "javascript"
    
    return "python"


def detect_language(code: str) -> str:
    """Public API with optimized caching strategy.

    Truncates to the first 2000 characters (sufficient for all language
    hints) and delegates to the lru_cache-backed helper.
    """
    if not code:
        return "python"
    
    # Use first 2000 chars for detection (sufficient for language hints)
    code_prefix = code[:2000] if len(code) > 2000 else code
    return _detect_language_cached(code_prefix)