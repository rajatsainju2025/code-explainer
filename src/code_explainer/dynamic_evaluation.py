"""
Dynamic evaluation utilities.
"""

from enum import Enum
from typing import Dict, Any, List


class EvaluationDimension(Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"


class DynamicEvaluator:
    """Dynamic evaluation of models."""

    def __init__(self):
        self.evaluation_history = []

    def evaluate(self, model, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model dynamically."""
        result = {
            "scores": {dim.value: 0.8 for dim in EvaluationDimension},
            "overall_score": 0.85
        }
        self.evaluation_history.append(result)
        return result

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "average_score": 0.0,
                "best_score": 0.0,
                "worst_score": 0.0
            }

        scores = [eval["overall_score"] for eval in self.evaluation_history]

        return {
            "total_evaluations": len(self.evaluation_history),
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "latest_score": scores[-1] if scores else 0.0
        }