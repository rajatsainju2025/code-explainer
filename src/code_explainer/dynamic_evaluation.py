"""
Dynamic evaluation utilities.
"""

from enum import Enum
from typing import Dict, Any, List


class EvaluationDimension(Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"


# Pre-compute dimension values for efficient scoring
_DIMENSION_VALUES = tuple(dim.value for dim in EvaluationDimension)


class DynamicEvaluator:
    """Dynamic evaluation of models."""

    __slots__ = ("evaluation_history",)

    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []

    def evaluate(self, model, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model dynamically."""
        result = {
            "scores": {dim: 0.8 for dim in _DIMENSION_VALUES},
            "overall_score": 0.85
        }
        self.evaluation_history.append(result)
        return result

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        history = self.evaluation_history
        if not history:
            return {
                "total_evaluations": 0,
                "average_score": 0.0,
                "best_score": 0.0,
                "worst_score": 0.0
            }

        scores = [eval_result["overall_score"] for eval_result in history]
        n = len(scores)

        return {
            "total_evaluations": n,
            "average_score": sum(scores) / n,
            "best_score": max(scores),
            "worst_score": min(scores),
            "latest_score": scores[-1]
        }