"""
Intelligent explanation generator for adaptive, audience-aware explanations.
"""

from enum import Enum
from dataclasses import dataclass


class ExplanationAudience(Enum):
    """Target audience for explanations."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class ExplanationStyle(Enum):
    """Style of explanation delivery."""
    CONCISE = "concise"
    DETAILED = "detailed"
    STEP_BY_STEP = "step_by_step"


@dataclass
class EnhancedExplanation:
    """Enhanced explanation with metadata."""
    explanation: str
    audience: ExplanationAudience
    style: ExplanationStyle
    confidence: float
    complexity_score: float
    key_concepts: list[str]


class IntelligentExplanationGenerator:
    """Generates intelligent, adaptive explanations based on audience and context."""

    def __init__(self):
        self.audience_profiles = {
            ExplanationAudience.BEGINNER: {
                "vocabulary_level": "simple",
                "technical_depth": "low",
                "examples_needed": True
            },
            ExplanationAudience.INTERMEDIATE: {
                "vocabulary_level": "moderate",
                "technical_depth": "medium",
                "examples_needed": True
            },
            ExplanationAudience.EXPERT: {
                "vocabulary_level": "advanced",
                "technical_depth": "high",
                "examples_needed": False
            }
        }

    def generate_explanation(
        self,
        code: str,
        base_explanation: str,
        audience: ExplanationAudience = ExplanationAudience.INTERMEDIATE,
        style: ExplanationStyle = ExplanationStyle.CONCISE
    ) -> EnhancedExplanation:
        """Generate an enhanced explanation."""
        # Simple implementation - in real system this would use ML models
        enhanced_text = self._adapt_for_audience(base_explanation, audience)
        enhanced_text = self._apply_style(enhanced_text, style)

        return EnhancedExplanation(
            explanation=enhanced_text,
            audience=audience,
            style=style,
            confidence=0.8,
            complexity_score=self._calculate_complexity(code),
            key_concepts=self._extract_key_concepts(code)
        )

    def _adapt_for_audience(self, explanation: str, audience: ExplanationAudience) -> str:
        """Adapt explanation for specific audience."""
        profile = self.audience_profiles[audience]

        if audience == ExplanationAudience.BEGINNER:
            explanation = explanation.replace("function", "mini-program")
            explanation = explanation.replace("algorithm", "step-by-step process")
        elif audience == ExplanationAudience.EXPERT:
            explanation = explanation.replace("This function", "The function")
            explanation = explanation.replace("It does", "It implements")

        return explanation

    def _apply_style(self, explanation: str, style: ExplanationStyle) -> str:
        """Apply explanation style."""
        if style == ExplanationStyle.STEP_BY_STEP:
            return f"Let's break this down step by step:\n\n{explanation}"
        elif style == ExplanationStyle.DETAILED:
            return f"Detailed Analysis:\n\n{explanation}"
        else:
            return explanation

    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity score."""
        # Use count() instead of split() for better performance
        line_count = code.count('\n') + 1
        return min(line_count / 50.0, 1.0)  # Simple heuristic

    def _extract_key_concepts(self, code: str) -> list[str]:
        """Extract key programming concepts from code."""
        concepts = []
        if "def " in code:
            concepts.append("function")
        if "class " in code:
            concepts.append("class")
        if "for " in code or "while " in code:
            concepts.append("loop")
        if "if " in code:
            concepts.append("conditional")
        return concepts