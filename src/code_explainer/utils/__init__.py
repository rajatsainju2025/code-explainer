"""Utilities module."""

from .ast_analysis import (
    ASTInfo,
    analyze_code_comprehensive,
    summarize_code_structure,
)
from .config import load_config, setup_logging
def get_device(*args, **kwargs):
    """Lazy wrapper that delegates to `utils.device.get_device` to avoid
    importing device-heavy modules at package import time."""
    from .device import get_device as _get_device
    return _get_device(*args, **kwargs)
from .hashing import fast_hash_bytes, fast_hash_str, json_loads, json_dumps
from .language import detect_language
from .prompting import prompt_for_language

# Internal helpers re-exported for backward compatibility
from .ast_analysis import (
    _collect_docstrings_from_code,
    _collect_import_docs,
    _extract_python_ast_info,
    _summarize_python_ast,
)
from .execution import _safe_exec_subprocess

__all__ = [
    # Public API
    "ASTInfo",
    "analyze_code_comprehensive",
    "detect_language",
    "fast_hash_bytes",
    "fast_hash_str",
    "get_device",
    "json_loads",
    "json_dumps",
    "load_config",
    "prompt_for_language",
    "setup_logging",
    "summarize_code_structure",
]