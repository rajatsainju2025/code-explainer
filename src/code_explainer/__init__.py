"""
Code Explainer: An efficient LLM-powered tool for explaining code.

This package provides tools for training and using language models
to generate human-readable explanations of code snippets.
"""

__version__ = "0.4.0"
__author__ = "Rajat Sainju"
__email__ = "your.email@example.com"

from .model import CodeExplainer
from .utils import load_config, setup_logging

# Optional imports
try:
    from .trainer import CodeExplainerTrainer
except ImportError:
    CodeExplainerTrainer = None

__all__ = [
    "CodeExplainer",
    "load_config",
    "setup_logging",
]

if CodeExplainerTrainer is not None:
    __all__.append("CodeExplainerTrainer")
