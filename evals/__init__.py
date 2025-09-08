"""
Unified evaluation framework for Code Explainer.

This module provides a comprehensive evaluation system with:
- Reproducible experiments with seed control
- Standardized metrics (accuracy, retrieval, latency, cost)
- Statistical analysis with confidence intervals
- Flexible configuration system
- JSON/CSV export for systematic comparison
"""

from .runner import EvalRunner
from .metrics import MetricsCalculator, EvalResults
from .config import EvalConfig, load_config
from .datasets import DatasetLoader
from .statistical import StatisticalAnalyzer

__version__ = "1.0.0"
__all__ = [
    "EvalRunner",
    "MetricsCalculator", 
    "EvalResults",
    "EvalConfig",
    "load_config",
    "DatasetLoader",
    "StatisticalAnalyzer"
]
