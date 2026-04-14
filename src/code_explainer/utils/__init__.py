"""Utilities module — public API surface only.

Internal helpers (prefixed with _) are accessible via their submodules
but are not re-exported here to keep the public API clean.
"""

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

__all__ = [
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