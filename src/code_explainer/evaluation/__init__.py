"""Advanced evaluation framework for Code Explainer.

This package implements state-of-the-art evaluation methods including:
- Traditional metrics (BLEU, ROUGE, BERTScore, CodeBLEU)
- LLM-as-a-Judge evaluation with multi-judge consensus
- Preference-based evaluation with Bradley-Terry ranking
- Contamination detection for data integrity
- Robustness testing with adversarial transformations

Usage:
    from code_explainer.evaluation import (
        run_contamination_detection,
        run_robustness_tests
    )
"""

from .contamination import run_contamination_detection
from .robustness import run_robustness_tests

__all__ = [
    "run_contamination_detection",
    "run_robustness_tests"
]

__version__ = "1.0.0"
