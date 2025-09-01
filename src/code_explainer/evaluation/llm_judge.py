"""LLM-as-a-Judge evaluation framework for code explanations.

This module implements state-of-the-art judge-based evaluation using multiple LLMs
to assess explanation quality across various dimensions.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class JudgmentCriteria:
    """Criteria for evaluating explanations."""
    
    name: str
    description: str
    scale: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    weight: float = 1.0


@dataclass
class JudgeResult:
    """Result from a single judge evaluation."""
    
    judge_model: str
    criteria_scores: Dict[str, float]
    overall_score: float
    reasoning: str
    confidence: float = 1.0
    latency_ms: int = 0


@dataclass
class MultiJudgeResult:
    """Result from multiple judge evaluation."""
    
    individual_results: List[JudgeResult]
    consensus_scores: Dict[str, float]
    agreement_metrics: Dict[str, float]
    final_score: float


class LLMJudge:
    """Base class for LLM judges."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
    
    def evaluate(
        self,
        code: str,
        explanation: str,
        reference: Optional[str] = None,
        criteria: Optional[List[JudgmentCriteria]] = None,
    ) -> JudgeResult:
        """Evaluate an explanation using this judge."""
        raise NotImplementedError
    
    def _prepare_prompt(
        self,
        code: str,
        explanation: str,
        reference: Optional[str] = None,
        criteria: Optional[List[JudgmentCriteria]] = None,
    ) -> str:
        """Prepare the evaluation prompt."""
        criteria = criteria or self._default_criteria()
        
        prompt = f"""You are an expert code reviewer tasked with evaluating the quality of a code explanation.

**Code to Explain:**
```python
{code}
```

**Generated Explanation:**
{explanation}
"""
        
        if reference:
            prompt += f"""
**Reference Explanation:**
{reference}
"""
        
        prompt += """
**Evaluation Criteria:**
"""
        
        for criterion in criteria:
            prompt += f"""
{criterion.name} ({criterion.weight:.1f}x weight): {criterion.description}
Scale: {min(criterion.scale)} (poor) to {max(criterion.scale)} (excellent)
"""
        
        prompt += """
**Instructions:**
1. Evaluate the explanation against each criterion
2. Provide a score for each criterion on the specified scale
3. Give a brief reasoning for your scores
4. Provide an overall weighted score

**Output Format (JSON):**
{
    "criteria_scores": {
        "criterion_name": score,
        ...
    },
    "reasoning": "Brief explanation of your evaluation",
    "overall_score": weighted_average_score,
    "confidence": confidence_level_0_to_1
}
"""
        
        return prompt
    
    def _default_criteria(self) -> List[JudgmentCriteria]:
        """Default evaluation criteria."""
        return [
            JudgmentCriteria(
                name="technical_accuracy",
                description="Technical correctness and accuracy of the explanation",
                weight=0.4
            ),
            JudgmentCriteria(
                name="clarity",
                description="Clarity, readability, and understandability",
                weight=0.3
            ),
            JudgmentCriteria(
                name="completeness",
                description="Coverage of important code aspects and concepts",
                weight=0.3
            )
        ]


class OpenAIJudge(LLMJudge):
    """OpenAI GPT-based judge."""
    
    def __init__(self, model_name: str = "gpt-4-turbo", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self._client = None
    
    def _get_client(self):
        """Get OpenAI client, importing if needed."""
        if self._client is None:
            try:
                import openai  # type: ignore[import-not-found]
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required for OpenAI judge")
        return self._client
    
    def evaluate(
        self,
        code: str,
        explanation: str,
        reference: Optional[str] = None,
        criteria: Optional[List[JudgmentCriteria]] = None,
    ) -> JudgeResult:
        """Evaluate using OpenAI API."""
        start_time = time.time()
        
        prompt = self._prepare_prompt(code, explanation, reference, criteria)
        
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.get("temperature", 0.0),
                max_tokens=self.config.get("max_tokens", 1000),
            )
            
            content = response.choices[0].message.content
            result_data = json.loads(content)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return JudgeResult(
                judge_model=self.model_name,
                criteria_scores=result_data["criteria_scores"],
                overall_score=result_data["overall_score"],
                reasoning=result_data["reasoning"],
                confidence=result_data.get("confidence", 1.0),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"OpenAI judge evaluation failed: {e}")
            return JudgeResult(
                judge_model=self.model_name,
                criteria_scores={},
                overall_score=0.0,
                reasoning=f"Evaluation failed: {e}",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000)
            )


class AnthropicJudge(LLMJudge):
    """Anthropic Claude-based judge."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self._client = None
    
    def _get_client(self):
        """Get Anthropic client, importing if needed."""
        if self._client is None:
            try:
                import anthropic  # type: ignore[import-not-found]
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required for Anthropic judge")
        return self._client
    
    def evaluate(
        self,
        code: str,
        explanation: str,
        reference: Optional[str] = None,
        criteria: Optional[List[JudgmentCriteria]] = None,
    ) -> JudgeResult:
        """Evaluate using Anthropic API."""
        start_time = time.time()
        
        prompt = self._prepare_prompt(code, explanation, reference, criteria)
        
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.config.get("max_tokens", 1000),
                temperature=self.config.get("temperature", 0.0),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            result_data = json.loads(content)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return JudgeResult(
                judge_model=self.model_name,
                criteria_scores=result_data["criteria_scores"],
                overall_score=result_data["overall_score"],
                reasoning=result_data["reasoning"],
                confidence=result_data.get("confidence", 1.0),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Anthropic judge evaluation failed: {e}")
            return JudgeResult(
                judge_model=self.model_name,
                criteria_scores={},
                overall_score=0.0,
                reasoning=f"Evaluation failed: {e}",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000)
            )


class MultiJudgeEvaluator:
    """Evaluator using multiple LLM judges for consensus."""
    
    def __init__(self, judges: List[LLMJudge]):
        self.judges = judges
    
    def evaluate(
        self,
        code: str,
        explanation: str,
        reference: Optional[str] = None,
        criteria: Optional[List[JudgmentCriteria]] = None,
        consensus_method: str = "average",
    ) -> MultiJudgeResult:
        """Evaluate using multiple judges."""
        individual_results = []
        
        for judge in self.judges:
            try:
                result = judge.evaluate(code, explanation, reference, criteria)
                individual_results.append(result)
            except Exception as e:
                logger.error(f"Judge {judge.model_name} failed: {e}")
        
        if not individual_results:
            raise ValueError("All judges failed to provide evaluations")
        
        # Compute consensus scores
        consensus_scores = self._compute_consensus(individual_results, consensus_method)
        
        # Compute agreement metrics
        agreement_metrics = self._compute_agreement(individual_results)
        
        # Compute final score
        final_score = consensus_scores.get("overall", 0.0)
        
        return MultiJudgeResult(
            individual_results=individual_results,
            consensus_scores=consensus_scores,
            agreement_metrics=agreement_metrics,
            final_score=final_score
        )
    
    def _compute_consensus(
        self,
        results: List[JudgeResult],
        method: str = "average"
    ) -> Dict[str, float]:
        """Compute consensus scores across judges."""
        if not results:
            return {}
        
        # Collect all criteria
        all_criteria = set()
        for result in results:
            all_criteria.update(result.criteria_scores.keys())
        
        consensus = {}
        
        if method == "average":
            # Simple average
            for criterion in all_criteria:
                scores = [r.criteria_scores.get(criterion, 0.0) for r in results if criterion in r.criteria_scores]
                if scores:
                    consensus[criterion] = sum(scores) / len(scores)
            
            overall_scores = [r.overall_score for r in results]
            consensus["overall"] = sum(overall_scores) / len(overall_scores)
        
        elif method == "median":
            # Median consensus
            import statistics
            for criterion in all_criteria:
                scores = [r.criteria_scores.get(criterion, 0.0) for r in results if criterion in r.criteria_scores]
                if scores:
                    consensus[criterion] = statistics.median(scores)
            
            overall_scores = [r.overall_score for r in results]
            consensus["overall"] = statistics.median(overall_scores)
        
        elif method == "majority":
            # Majority voting (for discrete scores)
            for criterion in all_criteria:
                scores = [r.criteria_scores.get(criterion, 0.0) for r in results if criterion in r.criteria_scores]
                if scores:
                    # Find most common score
                    consensus[criterion] = Counter(scores).most_common(1)[0][0]
            
            overall_scores = [r.overall_score for r in results]
            consensus["overall"] = Counter(overall_scores).most_common(1)[0][0]
        
        return consensus
    
    def _compute_agreement(self, results: List[JudgeResult]) -> Dict[str, float]:
        """Compute inter-judge agreement metrics."""
        if len(results) < 2:
            return {"agreement": 1.0}
        
        # Collect all criteria
        all_criteria = set()
        for result in results:
            all_criteria.update(result.criteria_scores.keys())
        
        agreements = {}
        
        # Compute pairwise correlation for each criterion
        for criterion in all_criteria:
            scores = [r.criteria_scores.get(criterion, 0.0) for r in results if criterion in r.criteria_scores]
            if len(scores) >= 2:
                # Compute variance as inverse of agreement
                import statistics
                if len(set(scores)) > 1:
                    variance = statistics.variance(scores)
                    # Convert to agreement score (0-1, higher is better)
                    agreements[criterion] = max(0.0, 1.0 - variance / 4.0)  # Assume max variance of 4
                else:
                    agreements[criterion] = 1.0  # Perfect agreement
        
        # Overall agreement
        if agreements:
            agreements["overall"] = sum(agreements.values()) / len(agreements)
        else:
            agreements["overall"] = 0.0
        
        return agreements


def create_judge_from_config(config: Dict[str, Any]) -> LLMJudge:
    """Create a judge from configuration."""
    model_name = config["model"]
    judge_type = config.get("type", "openai")
    api_key = config.get("api_key")
    params = config.get("params", {})
    
    if judge_type == "openai" or model_name.startswith("gpt"):
        return OpenAIJudge(model_name, api_key=api_key, **params)
    elif judge_type == "anthropic" or model_name.startswith("claude"):
        return AnthropicJudge(model_name, api_key=api_key, **params)
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


def load_criteria_from_file(criteria_file: Union[str, Path]) -> List[JudgmentCriteria]:
    """Load evaluation criteria from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for loading criteria from file")
    
    with open(criteria_file, 'r') as f:
        criteria_data = yaml.safe_load(f)
    
    criteria = []
    for name, config in criteria_data.get("criteria", {}).items():
        criteria.append(JudgmentCriteria(
            name=name,
            description=config["description"],
            scale=config.get("scale", [1, 2, 3, 4, 5]),
            weight=config.get("weight", 1.0)
        ))
    
    return criteria


def evaluate_with_judges(
    predictions_file: Union[str, Path],
    judges_config: Dict[str, Any],
    criteria: Optional[List[JudgmentCriteria]] = None,
    output_file: Optional[Union[str, Path]] = None,
    consensus_method: str = "average",
) -> List[MultiJudgeResult]:
    """Evaluate predictions using multiple judges."""
    # Load predictions (support JSONL or JSON array)
    p = Path(predictions_file)
    with p.open('r') as f:
        if p.suffix.lower() == '.jsonl':
            predictions = [json.loads(line) for line in f if line.strip()]
        else:
            predictions = json.load(f)
    
    # Create judges
    judges = []
    for judge_name, judge_config in judges_config.items():
        judge = create_judge_from_config(judge_config)
        judges.append(judge)

    # Set default criteria if none provided
    if criteria is None:
        criteria = [
            JudgmentCriteria(
                name="accuracy",
                description="How accurate and correct is the explanation?",
            ),
            JudgmentCriteria(
                name="clarity",
                description="How clear and understandable is the explanation?",
            ),
            JudgmentCriteria(
                name="completeness",
                description="How complete and comprehensive is the explanation?",
            ),
        ]

    # Evaluate all predictions
    results = []
    evaluator = MultiJudgeEvaluator(judges)
    for prediction in predictions:
        result = evaluator.evaluate(
            code=prediction.get("code", ""),
            # Treat model output as the explanation being judged
            explanation=prediction.get("prediction", prediction.get("explanation", "")),
            # Prefer an explicit reference field; fall back to ground-truth explanation
            reference=prediction.get("reference", prediction.get("explanation", "")),
            criteria=criteria,
            consensus_method=consensus_method,
        )
        results.append(result)

    # Save results if output file specified
    if output_file:
        save_results(results, output_file)

    return results


def run_llm_judge_evaluation(
    test_file: Union[str, Path],
    predictions_file: Union[str, Path],
    output_file: Union[str, Path],
    judges: Optional[List[str]] = None,
    criteria: Optional[List[str]] = None,
    require_consensus: bool = False
) -> Dict[str, Any]:
    """Convenience function for running LLM judge evaluation."""
    judges = judges or ["gpt-4", "claude-3-sonnet"]
    criteria_names = criteria or ["accuracy", "clarity", "completeness"]
    
    # Create judges config
    judges_config = {}
    import os
    for judge_name in judges:
        if judge_name.startswith("gpt"):
            judges_config[judge_name] = {
                "type": "openai",
                "model": judge_name,
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }
        elif judge_name.startswith("claude"):
            judges_config[judge_name] = {
                "type": "anthropic", 
                "model": judge_name,
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            }
    
    # Create criteria objects
    criteria_objects = []
    for criterion_name in criteria_names:
        if criterion_name == "accuracy":
            desc = "How accurate and correct is the explanation?"
        elif criterion_name == "clarity":
            desc = "How clear and understandable is the explanation?"
        elif criterion_name == "completeness":
            desc = "How complete and comprehensive is the explanation?"
        else:
            desc = f"Quality assessment for {criterion_name}"
        
        criteria_objects.append(JudgmentCriteria(name=criterion_name, description=desc))
    
    # Run evaluation
    results = evaluate_with_judges(
        predictions_file=predictions_file,
        judges_config=judges_config,
        criteria=criteria_objects,
        output_file=output_file,
        consensus_method="majority" if require_consensus else "average"
    )
    
    # Calculate summary statistics
    if results:
        overall_scores = [r.final_score for r in results]
        agreement_scores = [r.agreement_metrics.get("overall", 0.0) for r in results]
        
        return {
            "overall_score": sum(overall_scores) / len(overall_scores),
            "judge_agreement": sum(agreement_scores) / len(agreement_scores),
            "total_evaluations": len(results)
        }
    
    return {"overall_score": 0.0, "judge_agreement": 0.0, "total_evaluations": 0}


def save_results(results: List[MultiJudgeResult], output_file: Union[str, Path]):
    """Save evaluation results to a JSONL file (one result per line)."""
    if not output_file:
        return
    with open(output_file, 'w') as f:
        for result in results:
            result_dict = {
                "consensus_scores": result.consensus_scores,
                "agreement_metrics": result.agreement_metrics,
                "final_score": result.final_score,
                "individual_results": [
                    {
                        "judge_model": r.judge_model,
                        "criteria_scores": r.criteria_scores,
                        "overall_score": r.overall_score,
                        "reasoning": r.reasoning,
                        "confidence": r.confidence,
                        "latency_ms": r.latency_ms,
                    }
                    for r in result.individual_results
                ],
            }
            f.write(json.dumps(result_dict) + "\n")
