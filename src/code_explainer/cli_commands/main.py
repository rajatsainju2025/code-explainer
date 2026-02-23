"""Main CLI interface for code explainer."""

import gc

import click

from ..utils import setup_logging


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