"""
Code Explainer: An efficient LLM-powered tool for explaining code.

This package provides tools for training and using language models
to generate human-readable explanations of code snippets.
"""

__version__ = "0.6.0"
__author__ = "Rajat Sainju"
__email__ = "rajatsainju@proton.me"

from .model import CodeExplainer
from .utils import load_config, setup_logging
from .retrieval.model_cache import clear_model_cache, get_model_cache_info
from .cache_ttl import CacheTTLConfig, TTLCache, ONE_HOUR, TWO_HOURS, ONE_DAY

# Optional imports
try:
    from .trainer import CodeExplainerTrainer
except ImportError:
    CodeExplainerTrainer = None  # type: ignore[assignment,misc]

__all__ = [
    # Core
    "CodeExplainer",
    # Config helpers
    "load_config",
    "setup_logging",
    # Cache helpers
    "clear_model_cache",
    "get_model_cache_info",
    # Cache primitives
    "CacheTTLConfig",
    "TTLCache",
    "ONE_HOUR",
    "TWO_HOURS",
    "ONE_DAY",
    # Version metadata
    "__version__",
    "__author__",
    "__email__",
]

if CodeExplainerTrainer is not None:
    __all__.append("CodeExplainerTrainer")
