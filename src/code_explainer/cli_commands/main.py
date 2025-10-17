"""Main CLI interface for code explainer."""

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..model import CodeExplainer
from ..utils import setup_logging

console = Console()


@click.group(
    help="Code Explainer CLI - Train and use LLM models for code explanation.\n\nAliases: cx-train, cx-serve, cx-explain, cx-explain-file"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose):
    """Code Explainer CLI - Train and use LLM models for code explanation."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)


# Import and register command modules
from .commands import train, explain, serve

# Register commands
train.register_train_commands(main)
explain.register_explain_commands(main)
serve.register_serve_commands(main)