"""Main CLI interface for code explainer."""

import logging
import gc
from pathlib import Path
from functools import lru_cache

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..model import CodeExplainer
from ..utils import setup_logging

console = Console()


@lru_cache(maxsize=4)
def _get_cached_model(model_path, config_path):
    """Cache loaded models to reduce memory overhead."""
    return CodeExplainer(model_path=model_path, config_path=config_path)


@click.group(
    help="Code Explainer CLI - Train and use LLM models for code explanation.\n\nAliases: cx-train, cx-serve, cx-explain, cx-explain-file"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--optimize-memory", is_flag=True, help="Enable memory optimization (aggressive GC)")
def main(verbose, optimize_memory):
    """Code Explainer CLI - Train and use LLM models for code explanation."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)
    
    # Enable memory optimization if requested
    if optimize_memory:
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
        gc.enable()


# Import and register command modules
from .commands import train, explain, serve

# Register commands
train.register_train_commands(main)
explain.register_explain_commands(main)
serve.register_serve_commands(main)