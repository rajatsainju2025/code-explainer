"""Language detection utilities with fast substring heuristics."""

from functools import lru_cache

# Pre-compiled patterns and sets for fast detection
_CPP_PATTERNS = frozenset(['#include', 'std::', 'using namespace'])
_JAVA_PATTERNS = frozenset(['public static void main', 'class ', 'system.out'])
_JS_PATTERNS = frozenset(['function ', '=>', 'console.log'])


@lru_cache(maxsize=2048)
def _detect_language_cached(code: str) -> str:
    """Very simple language detector for code snippets - O(1) pattern matching.
    Returns one of: python, javascript, java, cpp.
    """
    code_l = code.lower()
    
    # Check each language with optimized pattern matching
    if any(p in code_l for p in _CPP_PATTERNS):
        return "cpp"
    if any(p in code_l for p in _JAVA_PATTERNS):
        return "java"
    if any(p in code_l for p in _JS_PATTERNS):
        return "javascript"
    return "python"


def detect_language(code: str) -> str:
    """Public API with small LRU cache to avoid repeated lowercase scans."""
    # Guard very large strings to avoid caching huge objects
    if len(code) > 10000:
        return _detect_language_cached.__wrapped__(code)  # type: ignore[attr-defined]
    return _detect_language_cached(code)