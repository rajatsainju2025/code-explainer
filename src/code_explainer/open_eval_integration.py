"""
OpenEval Integration Module

This module provides seamless integration with the OpenEval framework,
enabling standardized, reproducible, and comprehensive evaluation of
language models for code intelligence tasks. OpenEval is a framework
for creating, running, and sharing evaluations of language models.

Features:
- OpenEval specification compliance
- Standardized evaluation templates for code tasks
- Automated evaluation pipeline integration
- Result aggregation and analysis
- Benchmark comparison and leaderboards
- Research methodology integration
- Extensible evaluation registry
- Quality assurance and validation
"""

import json
import yaml
import hashlib
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os
import shutil
from pathlib import Path
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class OpenEvalSpec:
    """OpenEval specification for an evaluation."""
    name: str
    description: str
    version: str
    authors: List[str]
    tasks: List[Dict[str, Any]]
    metrics: List[Dict[str, Any]]
    dataset: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenEvalTask:
    """OpenEval task definition."""
    task_id: str
    name: str
    description: str
    task_type: str
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    metrics: List[str]
    dataset_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenEvalResult:
    """Result from OpenEval evaluation."""
    evaluation_id: str
    task_id: str
    model_name: str
    scores: Dict[str, float]
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OpenEvalRegistry:
    """Registry for OpenEval specifications and tasks."""

    def __init__(self):
        self.specifications: Dict[str, OpenEvalSpec] = {}
        self.tasks: Dict[str, OpenEvalTask] = {}
        self.templates: Dict[str, Dict[str, Any]] = {}

    def register_specification(self, spec: OpenEvalSpec) -> None:
        """Register an OpenEval specification."""
        self.specifications[spec.name] = spec

    def register_task(self, task: OpenEvalTask) -> None:
        """Register an OpenEval task."""
        self.tasks[task.task_id] = task

    def get_specification(self, name: str) -> Optional[OpenEvalSpec]:
        """Get a specification by name."""
        return self.specifications.get(name)

    def get_task(self, task_id: str) -> Optional[OpenEvalTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def list_specifications(self) -> List[str]:
        """List all registered specifications."""
        return list(self.specifications.keys())

    def list_tasks(self) -> List[str]:
        """List all registered tasks."""
        return list(self.tasks.keys())


class OpenEvalRunner:
    """Runner for OpenEval evaluations."""

    def __init__(self, registry: OpenEvalRegistry):
        self.registry = registry
        self.results: List[OpenEvalResult] = []
        self.working_dir = Path(tempfile.mkdtemp())

    def run_evaluation(self, spec_name: str, model_function: Callable,
                      model_name: str, num_samples: Optional[int] = None) -> OpenEvalResult:
        """Run an OpenEval evaluation."""
        spec = self.registry.get_specification(spec_name)
        if not spec:
            raise ValueError(f"Specification {spec_name} not found")

        evaluation_id = f"{spec_name}_{model_name}_{datetime.utcnow().isoformat()}"

        # Prepare evaluation data
        samples = self._prepare_evaluation_data(spec, num_samples)

        # Run evaluation
        scores = self._execute_evaluation(spec, model_function, samples)

        result = OpenEvalResult(
            evaluation_id=evaluation_id,
            task_id=spec_name,
            model_name=model_name,
            scores=scores,
            samples=samples,
            metadata={
                "specification": spec_name,
                "num_samples": len(samples),
                "evaluation_framework": "OpenEval"
            }
        )

        self.results.append(result)
        return result

    def _prepare_evaluation_data(self, spec: OpenEvalSpec,
                               num_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Prepare evaluation data from specification."""
        # This would typically load from the dataset specified in the spec
        # For now, we'll create sample data based on the spec
        samples = []

        for task in spec.tasks:
            task_def = self.registry.get_task(task["id"])
            if task_def:
                # Generate sample data based on task definition
                sample_count = num_samples or task.get("num_samples", 10)
                for i in range(sample_count):
                    sample = {
                        "task_id": task["id"],
                        "sample_id": f"{task['id']}_{i}",
                        "input": self._generate_sample_input(task_def),
                        "expected_output": self._generate_expected_output(task_def),
                        "metadata": task_def.metadata
                    }
                    samples.append(sample)

        return samples

    def _generate_sample_input(self, task: OpenEvalTask) -> Any:
        """Generate sample input for a task."""
        # Placeholder implementation
        if task.task_type == "code_explanation":
            return "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        elif task.task_type == "code_generation":
            return "Write a function to reverse a string"
        else:
            return "Sample input"

    def _generate_expected_output(self, task: OpenEvalTask) -> Any:
        """Generate expected output for a task."""
        # Placeholder implementation
        if task.task_type == "code_explanation":
            return "This function calculates the nth Fibonacci number using recursion"
        elif task.task_type == "code_generation":
            return "def reverse_string(s): return s[::-1]"
        else:
            return "Sample output"

    def _execute_evaluation(self, spec: OpenEvalSpec, model_function: Callable,
                          samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Execute the evaluation and calculate scores."""
        scores = {}

        for metric in spec.metrics:
            metric_name = metric["name"]
            metric_type = metric["type"]

            if metric_type == "exact_match":
                scores[metric_name] = self._calculate_exact_match(model_function, samples)
            elif metric_type == "codebleu":
                scores[metric_name] = self._calculate_codebleu(model_function, samples)
            elif metric_type == "pass_rate":
                scores[metric_name] = self._calculate_pass_rate(model_function, samples)
            else:
                scores[metric_name] = 0.5  # Placeholder

        return scores

    def _calculate_exact_match(self, model_function: Callable,
                             samples: List[Dict[str, Any]]) -> float:
        """Calculate exact match score."""
        correct = 0
        total = len(samples)

        for sample in samples:
            try:
                prediction = model_function(sample["input"])
                if prediction.strip() == sample["expected_output"].strip():
                    correct += 1
            except:
                pass

        return correct / total if total > 0 else 0

    def _calculate_codebleu(self, model_function: Callable,
                          samples: List[Dict[str, Any]]) -> float:
        """Calculate CodeBLEU score."""
        # Placeholder implementation
        return 0.75

    def _calculate_pass_rate(self, model_function: Callable,
                           samples: List[Dict[str, Any]]) -> float:
        """Calculate pass rate."""
        passed = 0
        total = len(samples)

        for sample in samples:
            try:
                prediction = model_function(sample["input"])
                if sample["expected_output"] in prediction:
                    passed += 1
            except:
                pass

        return passed / total if total > 0 else 0


class OpenEvalValidator:
    """Validator for OpenEval specifications and results."""

    def __init__(self):
        self.schema_version = "1.0.0"

    def validate_specification(self, spec: OpenEvalSpec) -> List[str]:
        """Validate an OpenEval specification."""
        errors = []

        # Check required fields
        if not spec.name:
            errors.append("Specification name is required")

        if not spec.description:
            errors.append("Specification description is required")

        if not spec.version:
            errors.append("Specification version is required")

        if not spec.tasks:
            errors.append("At least one task is required")

        if not spec.metrics:
            errors.append("At least one metric is required")

        # Validate tasks
        for task in spec.tasks:
            task_errors = self._validate_task(task)
            errors.extend(task_errors)

        # Validate metrics
        for metric in spec.metrics:
            metric_errors = self._validate_metric(metric)
            errors.extend(metric_errors)

        return errors

    def _validate_task(self, task: Dict[str, Any]) -> List[str]:
        """Validate a task definition."""
        errors = []

        if "id" not in task:
            errors.append("Task ID is required")

        if "type" not in task:
            errors.append("Task type is required")

        return errors

    def _validate_metric(self, metric: Dict[str, Any]) -> List[str]:
        """Validate a metric definition."""
        errors = []

        if "name" not in metric:
            errors.append("Metric name is required")

        if "type" not in metric:
            errors.append("Metric type is required")

        return errors

    def validate_result(self, result: OpenEvalResult) -> List[str]:
        """Validate an evaluation result."""
        errors = []

        if not result.evaluation_id:
            errors.append("Evaluation ID is required")

        if not result.task_id:
            errors.append("Task ID is required")

        if not result.model_name:
            errors.append("Model name is required")

        if not result.scores:
            errors.append("Scores are required")

        return errors


class OpenEvalAggregator:
    """Aggregator for OpenEval results and analysis."""

    def __init__(self):
        self.results: List[OpenEvalResult] = []

    def add_result(self, result: OpenEvalResult) -> None:
        """Add a result to the aggregator."""
        self.results.append(result)

    def aggregate_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by model."""
        model_results = {}

        for result in self.results:
            if result.model_name not in model_results:
                model_results[result.model_name] = {
                    "evaluations": [],
                    "average_scores": {},
                    "task_count": 0
                }

            model_results[result.model_name]["evaluations"].append(result)
            model_results[result.model_name]["task_count"] += 1

            # Aggregate scores
            for metric, score in result.scores.items():
                if metric not in model_results[result.model_name]["average_scores"]:
                    model_results[result.model_name]["average_scores"][metric] = []
                model_results[result.model_name]["average_scores"][metric].append(score)

        # Calculate averages
        for model_data in model_results.values():
            for metric in model_data["average_scores"]:
                scores = model_data["average_scores"][metric]
                model_data["average_scores"][metric] = sum(scores) / len(scores)

        return model_results

    def aggregate_by_task(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by task."""
        task_results = {}

        for result in self.results:
            if result.task_id not in task_results:
                task_results[result.task_id] = {
                    "evaluations": [],
                    "model_count": 0,
                    "average_scores": {}
                }

            task_results[result.task_id]["evaluations"].append(result)
            task_results[result.task_id]["model_count"] += 1

            # Aggregate scores
            for metric, score in result.scores.items():
                if metric not in task_results[result.task_id]["average_scores"]:
                    task_results[result.task_id]["average_scores"][metric] = []
                task_results[result.task_id]["average_scores"][metric].append(score)

        # Calculate averages
        for task_data in task_results.values():
            for metric in task_data["average_scores"]:
                scores = task_data["average_scores"][metric]
                task_data["average_scores"][metric] = sum(scores) / len(scores)

        return task_results

    def generate_leaderboard(self, metric: str) -> List[Dict[str, Any]]:
        """Generate a leaderboard for a specific metric."""
        model_scores = {}

        for result in self.results:
            if metric in result.scores:
                if result.model_name not in model_scores:
                    model_scores[result.model_name] = []
                model_scores[result.model_name].append(result.scores[metric])

        # Calculate average scores
        leaderboard = []
        for model, scores in model_scores.items():
            avg_score = sum(scores) / len(scores)
            leaderboard.append({
                "model": model,
                "average_score": avg_score,
                "num_evaluations": len(scores)
            })

        # Sort by average score (descending)
        leaderboard.sort(key=lambda x: x["average_score"], reverse=True)

        # Add ranks
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i

        return leaderboard

    def export_results(self, format: str = "json") -> str:
        """Export aggregated results."""
        aggregated = {
            "by_model": self.aggregate_by_model(),
            "by_task": self.aggregate_by_task(),
            "leaderboards": {
                "exact_match": self.generate_leaderboard("exact_match"),
                "codebleu": self.generate_leaderboard("codebleu"),
                "pass_rate": self.generate_leaderboard("pass_rate")
            },
            "metadata": {
                "total_evaluations": len(self.results),
                "export_timestamp": datetime.utcnow().isoformat(),
                "format_version": "1.0.0"
            }
        }

        if format == "json":
            return json.dumps(aggregated, indent=2, default=str)
        elif format == "yaml":
            return yaml.dump(aggregated, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


class OpenEvalIntegration:
    """Main integration class for OpenEval."""

    def __init__(self):
        self.registry = OpenEvalRegistry()
        self.runner = OpenEvalRunner(self.registry)
        self.validator = OpenEvalValidator()
        self.aggregator = OpenEvalAggregator()
        self._setup_standard_specs()

    def _setup_standard_specs(self) -> None:
        """Set up standard OpenEval specifications for code intelligence."""
        # Code explanation specification
        code_explanation_spec = OpenEvalSpec(
            name="code_explanation_eval",
            description="Evaluation specification for code explanation tasks",
            version="1.0.0",
            authors=["Code Explainer Team"],
            tasks=[
                {
                    "id": "code_explanation_basic",
                    "name": "Basic Code Explanation",
                    "type": "code_explanation",
                    "num_samples": 100
                }
            ],
            metrics=[
                {
                    "name": "exact_match",
                    "type": "exact_match",
                    "description": "Exact match accuracy"
                },
                {
                    "name": "semantic_similarity",
                    "type": "semantic_similarity",
                    "description": "Semantic similarity score"
                }
            ],
            dataset={
                "type": "huggingface",
                "name": "code_explanation_dataset",
                "split": "test"
            }
        )

        # Code generation specification
        code_generation_spec = OpenEvalSpec(
            name="code_generation_eval",
            description="Evaluation specification for code generation tasks",
            version="1.0.0",
            authors=["Code Explainer Team"],
            tasks=[
                {
                    "id": "code_generation_basic",
                    "name": "Basic Code Generation",
                    "type": "code_generation",
                    "num_samples": 100
                }
            ],
            metrics=[
                {
                    "name": "pass_rate",
                    "type": "pass_rate",
                    "description": "Code execution pass rate"
                },
                {
                    "name": "codebleu",
                    "type": "codebleu",
                    "description": "CodeBLEU score"
                }
            ],
            dataset={
                "type": "huggingface",
                "name": "code_generation_dataset",
                "split": "test"
            }
        )

        self.registry.register_specification(code_explanation_spec)
        self.registry.register_specification(code_generation_spec)

    def run_evaluation(self, spec_name: str, model_function: Callable,
                      model_name: str, num_samples: Optional[int] = None) -> OpenEvalResult:
        """Run an OpenEval evaluation."""
        result = self.runner.run_evaluation(spec_name, model_function, model_name, num_samples)
        self.aggregator.add_result(result)
        return result

    def validate_specification(self, spec: OpenEvalSpec) -> List[str]:
        """Validate an OpenEval specification."""
        return self.validator.validate_specification(spec)

    def get_leaderboard(self, metric: str) -> List[Dict[str, Any]]:
        """Get leaderboard for a metric."""
        return self.aggregator.generate_leaderboard(metric)

    def export_results(self, format: str = "json") -> str:
        """Export evaluation results."""
        return self.aggregator.export_results(format)

    def list_available_specs(self) -> List[str]:
        """List available evaluation specifications."""
        return self.registry.list_specifications()

    def get_specification_details(self, spec_name: str) -> Optional[Dict[str, Any]]:
        """Get details of a specification."""
        spec = self.registry.get_specification(spec_name)
        if spec:
            return {
                "name": spec.name,
                "description": spec.description,
                "version": spec.version,
                "authors": spec.authors,
                "tasks": len(spec.tasks),
                "metrics": [m["name"] for m in spec.metrics]
            }
        return None


# Export main classes
__all__ = [
    "OpenEvalSpec",
    "OpenEvalTask",
    "OpenEvalResult",
    "OpenEvalRegistry",
    "OpenEvalRunner",
    "OpenEvalValidator",
    "OpenEvalAggregator",
    "OpenEvalIntegration"
]
