"""Main CLI interface for code explainer."""

import gc
import time
from functools import lru_cache, wraps

import click
from rich.console import Console

from ..model import CodeExplainer
from ..utils import setup_logging

console = Console()
_cli_stats = {"start_time": None, "commands_run": 0, "total_time": 0.0}


@lru_cache(maxsize=4)
def _get_cached_model(model_path, config_path):
    """Cache loaded models to reduce memory overhead."""
    return CodeExplainer(model_path=model_path, config_path=config_path)


@click.group(
    help="Code Explainer CLI - Train and use LLM models for code explanation.\n\nAliases: cx-train, cx-serve, cx-explain, cx-explain-file"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--optimize-memory", is_flag=True, help="Enable memory optimization (aggressive GC)")
@click.option("--monitor-performance", is_flag=True, help="Enable performance monitoring and stats")
def main(verbose, optimize_memory, monitor_performance):
    """Code Explainer CLI - Train and use LLM models for code explanation."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)
    
    # Initialize performance tracking
    _cli_stats["start_time"] = time.time()
    _cli_stats["monitor"] = monitor_performance
    
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


def monitor_command(func):
    """Decorator to monitor command execution time and stats."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start
            _cli_stats["commands_run"] += 1
            _cli_stats["total_time"] += elapsed
            
            if _cli_stats.get("monitor"):
                console.print(f"\n⏱️  Command completed in {elapsed:.2f}s", style="cyan")
                if _cli_stats["commands_run"] > 1:
                    avg_time = _cli_stats["total_time"] / _cli_stats["commands_run"]
                    console.print(f"📊 Average time per command: {avg_time:.2f}s", style="cyan")
    
    return wrapper