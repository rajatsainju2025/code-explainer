"""
Main module entry point for the evaluation system.

Usage:
    python -m evals run --config configs/default.yaml
    python -m evals ablation --config configs/default.yaml --components retrieval
    python -m evals benchmark --suite standard
"""

from .cli import main

if __name__ == '__main__':
    main()
