"""Training entrypoint for Code Explainer using the package trainer."""

import argparse
import sys
from pathlib import Path

# Ensure package is importable when running from repo root
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_explainer.trainer import CodeExplainerTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Code Explainer model")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data",
        "-d",
        default=None,
        help="Optional path to JSON training data (overrides config)",
    )
    args = parser.parse_args()

    trainer = CodeExplainerTrainer(config_path=args.config)
    trainer.train(data_path=args.data)


if __name__ == "__main__":
    main()