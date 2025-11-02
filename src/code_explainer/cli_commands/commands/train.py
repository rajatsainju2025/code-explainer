"""Training commands for CLI."""

import click
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.panel import Panel

console = Console()
executor = ThreadPoolExecutor(max_workers=2)  # Limit to avoid resource exhaustion


def register_train_commands(main_group):
    """Register training-related commands."""

    @main_group.command()
    @click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
    @click.option("--data", "-d", help="Path to training data (JSON format)")
    @click.option("--async", "async_mode", is_flag=True, help="Run training in background async mode")
    def train(config, data, async_mode):
        """Train a new code explanation model."""
        console.print(Panel.fit("🚀 Starting Model Training", style="bold blue"))

        try:
            from ...trainer import CodeExplainerTrainer

            # Load config
            from ...utils.config import load_config
            config = load_config(config)
            
            trainer = CodeExplainerTrainer(config=config)
            
            if async_mode:
                # Run training in thread pool for async behavior
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_train_async(trainer, data))
                finally:
                    loop.close()
            else:
                trainer.train(train_dataset=data)
            
            console.print(Panel.fit("✅ Training completed successfully!", style="bold green"))
        except (ImportError, FileNotFoundError, RuntimeError, ValueError) as e:
            console.print(Panel.fit(f"❌ Training failed: {e}", style="bold red"))
            raise


async def _train_async(trainer, data_path):
    """Run training asynchronously in thread pool."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, partial(trainer.train, train_dataset=data_path))