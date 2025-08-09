"""
Code Explainer: An efficient LLM-powered tool for explaining code.

This package provides tools for training and using language models
to generate human-readable explanations of code snippets.
"""

__version__ = "0.2.2"
__author__ = "Rajat Sainju"
__email__ = "your.email@example.com"

from .model import CodeExplainer
from .trainer import CodeExplainerTrainer
from .utils import load_config, setup_logging

__all__ = [
    "CodeExplainer",
    "CodeExplainerTrainer", 
    "load_config",
    "setup_logging",
]
