"""Preference-based evaluation using pairwise comparisons and ranking models.

This module implements modern preference learning approaches for code explanation
evaluation, including Bradley-Terry ranking and constitutional AI principles.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize

from .llm_judge import JudgmentCriteria, LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class PairwiseComparison:
    """A single pairwise comparison result."""

    item_a: str
    item_b: str
    preference: float  # -1 (prefer B), 0 (tie), 1 (prefer A)
    confidence: float = 1.0
    reasoning: str = ""
    judge_model: str = ""


@dataclass
class RankingResult:
    """Result from ranking evaluation."""

    items: List[str]
    scores: List[float]  # Bradley-Terry scores
    rankings: List[int]  # 1-indexed rankings
    pairwise_comparisons: List[PairwiseComparison]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None


class PairwiseJudge:
    """Judge for pairwise preference evaluation."""

    def __init__(self, llm_judge: LLMJudge, criteria: Optional[List[JudgmentCriteria]] = None):
        self.llm_judge = llm_judge
        self.criteria = criteria or self._default_criteria()

    def compare(
        self,
        code: str,
        explanation_a: str,
        explanation_b: str,
        context: Optional[str] = None
    ) -> PairwiseComparison:
        """Compare two explanations and return preference."""
        prompt = self._prepare_pairwise_prompt(code, explanation_a, explanation_b, context)

        try:
            # Use the LLM judge with a custom evaluation
            result = self._evaluate_pairwise(prompt)

            return PairwiseComparison(
                item_a="explanation_a",
                item_b="explanation_b",
                preference=result["preference"],
                confidence=result.get("confidence", 1.0),
                reasoning=result.get("reasoning", ""),
                judge_model=self.llm_judge.model_name
            )

        except Exception as e:
            logger.error(f"Pairwise comparison failed: {e}")
            return PairwiseComparison(
                item_a="explanation_a",
                item_b="explanation_b",
                preference=0.0,  # Tie in case of error
                confidence=0.0,
                reasoning=f"Comparison failed: {e}",
                judge_model=self.llm_judge.model_name
            )

    def _prepare_pairwise_prompt(
        self,
        code: str,
        explanation_a: str,
        explanation_b: str,
        context: Optional[str] = None
    ) -> str:
        """Prepare prompt for pairwise comparison."""
        prompt = f"""You are an expert code reviewer tasked with comparing two explanations for the same code.

**Code to Explain:**
```python
{code}
```

**Explanation A:**
{explanation_a}

**Explanation B:**
{explanation_b}
"""

        if context:
            prompt += f"""
**Additional Context:**
{context}
"""

        prompt += f"""
**Evaluation Criteria:**
"""

        for criterion in self.criteria:
            prompt += f"""
- {criterion.name}: {criterion.description}
"""

        prompt += """
**Instructions:**
Compare the two explanations and determine which is better overall.
Consider all criteria and provide your reasoning.

**Output Format (JSON):**
{
    "preference": preference_score,  // -1.0 (strongly prefer B), -0.5 (slightly prefer B), 0.0 (tie), 0.5 (slightly prefer A), 1.0 (strongly prefer A)
    "reasoning": "Detailed comparison explaining your preference",
    "confidence": confidence_level_0_to_1,
    "criterion_analysis": {
        "criterion_name": "brief analysis for this criterion",
        ...
    }
}
"""

        return prompt

    def _evaluate_pairwise(self, prompt: str) -> Dict[str, Any]:
        """Evaluate pairwise comparison using LLM judge."""
        # For simplicity, we'll use a mock implementation
        # In practice, this would call the LLM judge with the prompt

        # Mock implementation - replace with actual LLM call
        return {
            "preference": 0.0,  # Neutral preference
            "reasoning": "Mock comparison - replace with actual LLM evaluation",
            "confidence": 0.5,
            "criterion_analysis": {}
        }

    def _default_criteria(self) -> List[JudgmentCriteria]:
        """Default criteria for pairwise comparison."""
        return [
            JudgmentCriteria(
                name="overall_quality",
                description="Overall quality considering accuracy, clarity, and completeness",
                weight=1.0
            )
        ]


class BradleyTerryRanker:
    """Bradley-Terry model for ranking items based on pairwise comparisons."""

    def __init__(self, regularization: float = 0.1):
        self.regularization = regularization

    def rank(
        self,
        items: List[str],
        comparisons: List[PairwiseComparison]
    ) -> RankingResult:
        """Rank items using Bradley-Terry model."""
        if len(items) < 2:
            raise ValueError("Need at least 2 items to rank")

        # Create item index mapping
        item_to_idx = {item: i for i, item in enumerate(items)}
        n_items = len(items)

        # Build comparison matrix
        wins = np.zeros((n_items, n_items))
        comparisons_count = np.zeros((n_items, n_items))

        for comp in comparisons:
            if comp.item_a not in item_to_idx or comp.item_b not in item_to_idx:
                continue

            idx_a = item_to_idx[comp.item_a]
            idx_b = item_to_idx[comp.item_b]

            comparisons_count[idx_a, idx_b] += 1
            comparisons_count[idx_b, idx_a] += 1

            if comp.preference > 0:  # A preferred
                wins[idx_a, idx_b] += abs(comp.preference)
            elif comp.preference < 0:  # B preferred
                wins[idx_b, idx_a] += abs(comp.preference)
            # If preference == 0, it's a tie, no wins added

        # Solve Bradley-Terry model
        scores = self._solve_bradley_terry(wins, comparisons_count)

        # Convert to rankings (1-indexed, lower is better)
        rankings = self._scores_to_rankings(scores)

        return RankingResult(
            items=items,
            scores=scores.tolist(),
            rankings=rankings,
            pairwise_comparisons=comparisons
        )

    def _solve_bradley_terry(
        self,
        wins: np.ndarray,
        comparisons: np.ndarray
    ) -> np.ndarray:
        """Solve Bradley-Terry model using maximum likelihood estimation."""
        n_items = wins.shape[0]

        def negative_log_likelihood(log_strengths):
            strengths = np.exp(log_strengths)
            nll = 0.0

            for i in range(n_items):
                for j in range(n_items):
                    if i != j and comparisons[i, j] > 0:
                        p_i_beats_j = strengths[i] / (strengths[i] + strengths[j])
                        # Add small epsilon to avoid log(0)
                        p_i_beats_j = max(p_i_beats_j, 1e-10)
                        nll -= wins[i, j] * math.log(p_i_beats_j)

            # Add L2 regularization
            nll += self.regularization * np.sum(log_strengths ** 2)

            return nll

        # Initialize with uniform strengths
        initial_log_strengths = np.zeros(n_items)

        # Optimize
        result = optimize.minimize(
            negative_log_likelihood,
            initial_log_strengths,
            method='BFGS'
        )

        if not result.success:
            logger.warning("Bradley-Terry optimization did not converge")

        # Convert back to strengths and normalize
        strengths = np.exp(result.x)
        scores = strengths / np.sum(strengths)

        return scores

    def _scores_to_rankings(self, scores: np.ndarray) -> List[int]:
        """Convert scores to rankings (1-indexed, lower is better)."""
        # Sort indices by score (descending)
        sorted_indices = np.argsort(-scores)

        # Create rankings
        rankings = np.zeros(len(scores), dtype=int)
        for rank, idx in enumerate(sorted_indices):
            rankings[idx] = rank + 1

        return rankings.tolist()


class PreferenceEvaluator:
    """Main class for preference-based evaluation."""

    def __init__(self, pairwise_judge: PairwiseJudge, ranker: Optional[BradleyTerryRanker] = None):
        self.pairwise_judge = pairwise_judge
        self.ranker = ranker or BradleyTerryRanker()

    def evaluate_preferences(
        self,
        code_examples: List[Dict[str, Any]],
        explanation_sets: List[List[str]],
        system_names: List[str]
    ) -> List[RankingResult]:
        """Evaluate preferences across multiple systems and examples."""
        results = []

        for i, (example, explanations) in enumerate(zip(code_examples, explanation_sets)):
            code = example.get("code", "")
            context = example.get("context", "")

            logger.info(f"Evaluating preferences for example {i+1}/{len(code_examples)}")

            # Generate all pairwise comparisons
            comparisons = []
            for j, k in itertools.combinations(range(len(explanations)), 2):
                comparison = self.pairwise_judge.compare(
                    code=code,
                    explanation_a=explanations[j],
                    explanation_b=explanations[k],
                    context=context
                )
                # Map back to system names
                comparison.item_a = system_names[j]
                comparison.item_b = system_names[k]
                comparisons.append(comparison)

            # Rank systems for this example
            ranking = self.ranker.rank(system_names, comparisons)
            results.append(ranking)

        return results

    def aggregate_rankings(self, ranking_results: List[RankingResult]) -> Dict[str, Any]:
        """Aggregate rankings across multiple examples."""
        if not ranking_results:
            return {}

        system_names = ranking_results[0].items
        n_systems = len(system_names)
        n_examples = len(ranking_results)

        # Collect all rankings
        all_rankings = np.array([result.rankings for result in ranking_results])
        all_scores = np.array([result.scores for result in ranking_results])

        # Compute aggregate statistics
        mean_rankings = np.mean(all_rankings, axis=0)
        mean_scores = np.mean(all_scores, axis=0)
        std_rankings = np.std(all_rankings, axis=0)

        # Count wins (how often each system ranked 1st)
        wins = np.sum(all_rankings == 1, axis=0)

        # Compute pairwise win rates
        pairwise_wins = np.zeros((n_systems, n_systems))
        for result in ranking_results:
            for comp in result.pairwise_comparisons:
                if comp.item_a in system_names and comp.item_b in system_names:
                    idx_a = system_names.index(comp.item_a)
                    idx_b = system_names.index(comp.item_b)

                    if comp.preference > 0:
                        pairwise_wins[idx_a, idx_b] += 1
                    elif comp.preference < 0:
                        pairwise_wins[idx_b, idx_a] += 1

        # Create summary
        summary = {
            "system_names": system_names,
            "n_examples": n_examples,
            "mean_rankings": mean_rankings.tolist(),
            "mean_scores": mean_scores.tolist(),
            "std_rankings": std_rankings.tolist(),
            "wins": wins.tolist(),
            "win_rates": (wins / n_examples).tolist(),
            "pairwise_win_matrix": pairwise_wins.tolist()
        }

        # Sort by mean ranking (lower is better)
        sorted_indices = np.argsort(mean_rankings)
        summary["leaderboard"] = [
            {
                "rank": i + 1,
                "system": system_names[idx],
                "mean_ranking": mean_rankings[idx],
                "mean_score": mean_scores[idx],
                "win_rate": wins[idx] / n_examples,
                "wins": int(wins[idx])
            }
            for i, idx in enumerate(sorted_indices)
        ]

        return summary


def load_constitutional_principles(principles_file: Union[str, Path]) -> List[JudgmentCriteria]:
    """Load constitutional AI principles from file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for loading constitutional principles")

    with open(principles_file, 'r') as f:
        principles_data = yaml.safe_load(f)

    criteria = []
    for name, config in principles_data.get("principles", {}).items():
        criteria.append(JudgmentCriteria(
            name=name,
            description=config["description"],
            scale=config.get("scale", [-2, -1, 0, 1, 2]),  # Allow negative preferences
            weight=config.get("weight", 1.0)
        ))

    return criteria


def run_preference_evaluation(
    predictions_files: List[Union[str, Path]],
    system_names: List[str],
    judge_config: Dict[str, Any],
    output_file: Union[str, Path],
    criteria_file: Optional[Union[str, Path]] = None,
    max_examples: Optional[int] = None
) -> Dict[str, Any]:
    """Run complete preference evaluation."""
    from .llm_judge import create_judge_from_config

    # Load predictions
    all_predictions = []
    for pred_file in predictions_files:
        with open(pred_file, 'r') as f:
            predictions = [json.loads(line) for line in f]
        all_predictions.append(predictions)

    # Ensure all files have same number of examples
    min_examples = min(len(preds) for preds in all_predictions)
    if max_examples:
        min_examples = min(min_examples, max_examples)

    # Create judge
    llm_judge = create_judge_from_config(judge_config)

    # Load criteria if provided
    criteria = None
    if criteria_file:
        criteria = load_constitutional_principles(criteria_file)

    pairwise_judge = PairwiseJudge(llm_judge, criteria)
    evaluator = PreferenceEvaluator(pairwise_judge)

    # Prepare data
    code_examples = []
    explanation_sets = []

    for i in range(min_examples):
        # Get code from first file (should be same across all)
        code_example = {
            "code": all_predictions[0][i].get("code", ""),
            "context": all_predictions[0][i].get("context", "")
        }
        code_examples.append(code_example)

        # Get explanations from all systems
        explanations = [
            preds[i].get("prediction", preds[i].get("explanation", ""))
            for preds in all_predictions
        ]
        explanation_sets.append(explanations)

    # Run evaluation
    logger.info(f"Running preference evaluation on {len(code_examples)} examples")
    ranking_results = evaluator.evaluate_preferences(
        code_examples, explanation_sets, system_names
    )

    # Aggregate results
    summary = evaluator.aggregate_rankings(ranking_results)

    # Save results
    with open(output_file, 'w') as f:
        json.dump({
            "summary": summary,
            "individual_rankings": [
                {
                    "items": result.items,
                    "scores": result.scores,
                    "rankings": result.rankings,
                    "pairwise_comparisons": [
                        {
                            "item_a": comp.item_a,
                            "item_b": comp.item_b,
                            "preference": comp.preference,
                            "confidence": comp.confidence,
                            "reasoning": comp.reasoning
                        }
                        for comp in result.pairwise_comparisons
                    ]
                }
                for result in ranking_results
            ]
        }, f, indent=2)

    return summary
