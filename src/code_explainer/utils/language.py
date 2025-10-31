"""Language detection utilities with fast substring heuristics."""

from functools import lru_cache


@lru_cache(maxsize=2048)
def _detect_language_cached(code: str) -> str:
    """Very simple language detector for code snippets.
    Returns one of: python, javascript, java, cpp.
    """
    code_l = code.lower()
    if "#include" in code_l or "std::" in code or ";" in code and "using namespace" in code_l:
        return "cpp"
    if "public static void main" in code_l or "class " in code and "system.out" in code_l:
        return "java"
    if "function " in code_l or "=>" in code or "console.log" in code_l:
        return "javascript"
    return "python"


def detect_language(code: str) -> str:
    """Public API with small LRU cache to avoid repeated lowercase scans."""
    # Guard very large strings to avoid caching huge objects
    if len(code) > 10000:
        return _detect_language_cached.__wrapped__(code)  # type: ignore[attr-defined]
    return _detect_language_cached(code)