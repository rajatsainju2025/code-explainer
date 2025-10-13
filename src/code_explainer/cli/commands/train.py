"""Training commands for CLI."""

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


def register_train_commands(main_group):
    """Register training-related commands."""

    @main_group.command()
    @click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
    @click.option("--data", "-d", help="Path to training data (JSON format)")
    def train(config, data):
        """Train a new code explanation model."""
        console.print(Panel.fit("🚀 Starting Model Training", style="bold blue"))

        try:
            from ..trainer import CodeExplainerTrainer

            trainer = CodeExplainerTrainer(config_path=config)
            trainer.train(data_path=data)
            console.print(Panel.fit("✅ Training completed successfully!", style="bold green"))
        except Exception as e:
            console.print(Panel.fit(f"❌ Training failed: {e}", style="bold red"))
            raise