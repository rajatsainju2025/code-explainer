"""Dynamic evaluation system that adapts to model capabilities."""

import asyncio
import json
import logging
import time
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels for dynamic evaluation."""
    TRIVIAL = 0.1
    EASY = 0.3
    MEDIUM = 0.5
    HARD = 0.7
    EXPERT = 0.9


class EvaluationDimension(Enum):
    """Dimensions of evaluation."""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    DOCUMENTATION = "documentation"


@dataclass
class ModelCapability:
    """Model capability metrics."""
    dimension: EvaluationDimension
    current_score: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    last_updated: datetime
    trend: str = "stable"  # "improving", "declining", "stable"


@dataclass
class DynamicTask:
    """A dynamically generated evaluation task."""
    task_id: str
    prompt: str
    expected_output: Optional[str]
    difficulty: DifficultyLevel
    dimensions: List[EvaluationDimension]
    generation_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResult:
    """Result of dynamic evaluation."""
    task_id: str
    model_response: str
    scores: Dict[EvaluationDimension, float]
    overall_score: float
    execution_time: float
    feedback: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class TaskGenerator:
    """Generates dynamic evaluation tasks."""
    
    def __init__(self, template_path: Optional[str] = None):
        """Initialize task generator.
        
        Args:
            template_path: Path to task templates
        """
        self.template_path = template_path
        self.templates = self._load_templates()
        self.used_combinations = set()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load task templates from file or use defaults."""
        if self.template_path and Path(self.template_path).exists():
            try:
                with open(self.template_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load templates: {e}")
        
        # Default templates
        return {
            "code_explanation": [
                {
                    "template": "Explain the following {language} function:\n\n```{language}\n{code}\n```\n\nProvide a clear explanation of what it does, how it works, and any potential issues.",
                    "variables": ["language", "code"],
                    "difficulty_factors": ["code_complexity", "language_obscurity", "algorithmic_complexity"]
                }
            ],
            "bug_detection": [
                {
                    "template": "Review this {language} code for bugs and security vulnerabilities:\n\n```{language}\n{code}\n```\n\nList all issues found and suggest fixes.",
                    "variables": ["language", "code"],
                    "difficulty_factors": ["bug_subtlety", "security_implications", "code_length"]
                }
            ],
            "optimization": [
                {
                    "template": "Analyze and optimize this {language} code for better performance:\n\n```{language}\n{code}\n```\n\nExplain the current time/space complexity and provide optimized version.",
                    "variables": ["language", "code"],
                    "difficulty_factors": ["algorithmic_complexity", "optimization_potential", "trade_offs"]
                }
            ]
        }
    
    def generate_code_sample(self, difficulty: DifficultyLevel, language: str = "python") -> str:
        """Generate a code sample based on difficulty.
        
        Args:
            difficulty: Target difficulty level
            language: Programming language
            
        Returns:
            Generated code sample
        """
        if language == "python":
            if difficulty == DifficultyLevel.TRIVIAL:
                samples = [
                    "def add(a, b):\n    return a + b",
                    "x = 5\nprint(x)",
                    "numbers = [1, 2, 3]\nprint(len(numbers))"
                ]
            elif difficulty == DifficultyLevel.EASY:
                samples = [
                    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                    "def is_even(num):\n    return num % 2 == 0\n\nfor i in range(10):\n    if is_even(i):\n        print(f'{i} is even')",
                    "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def subtract(self, a, b):\n        return a - b"
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                samples = [
                    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
                    "import threading\nimport time\n\nclass ThreadSafeCounter:\n    def __init__(self):\n        self._value = 0\n        self._lock = threading.Lock()\n    \n    def increment(self):\n        with self._lock:\n            self._value += 1\n    \n    def get_value(self):\n        with self._lock:\n            return self._value"
                ]
            elif difficulty == DifficultyLevel.HARD:
                samples = [
                    "import asyncio\nimport aiohttp\nfrom typing import List, Dict, Any\n\nclass AsyncDataProcessor:\n    def __init__(self, max_concurrent: int = 10):\n        self.semaphore = asyncio.Semaphore(max_concurrent)\n        self.session = None\n    \n    async def __aenter__(self):\n        self.session = aiohttp.ClientSession()\n        return self\n    \n    async def __aexit__(self, exc_type, exc_val, exc_tb):\n        if self.session:\n            await self.session.close()\n    \n    async def fetch_data(self, url: str) -> Dict[str, Any]:\n        async with self.semaphore:\n            async with self.session.get(url) as response:\n                return await response.json()\n    \n    async def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:\n        tasks = [self.fetch_data(url) for url in urls]\n        return await asyncio.gather(*tasks, return_exceptions=True)",
                    "from functools import wraps\nimport time\nfrom typing import Callable, Any\n\ndef retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):\n    def decorator(func: Callable) -> Callable:\n        @wraps(func)\n        async def wrapper(*args, **kwargs) -> Any:\n            for attempt in range(max_retries + 1):\n                try:\n                    return await func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_retries:\n                        raise e\n                    delay = min(base_delay * (2 ** attempt), max_delay)\n                    await asyncio.sleep(delay)\n        return wrapper\n    return decorator"
                ]
            else:  # EXPERT
                samples = [
                    "import ast\nimport inspect\nfrom typing import Dict, List, Any, Optional, Union\n\nclass CodeAnalyzer(ast.NodeVisitor):\n    def __init__(self):\n        self.metrics = {\n            'complexity': 0,\n            'functions': [],\n            'classes': [],\n            'imports': [],\n            'security_issues': []\n        }\n        self.current_function = None\n    \n    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:\n        self.current_function = node.name\n        func_info = {\n            'name': node.name,\n            'args': [arg.arg for arg in node.args.args],\n            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],\n            'complexity': self._calculate_complexity(node)\n        }\n        self.metrics['functions'].append(func_info)\n        self.generic_visit(node)\n        self.current_function = None\n    \n    def _calculate_complexity(self, node: ast.AST) -> int:\n        complexity = 1\n        for child in ast.walk(node):\n            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):\n                complexity += 1\n            elif isinstance(child, ast.BoolOp):\n                complexity += len(child.values) - 1\n        return complexity\n    \n    def _get_decorator_name(self, decorator: ast.AST) -> str:\n        if isinstance(decorator, ast.Name):\n            return decorator.id\n        elif isinstance(decorator, ast.Attribute):\n            return f'{decorator.value.id}.{decorator.attr}'\n        return 'unknown'"
                ]
            
            return random.choice(samples)
        
        # Add more languages as needed
        return f"// {language} code sample for difficulty {difficulty.name}"
    
    def generate_task(self, 
                     difficulty: DifficultyLevel,
                     dimensions: List[EvaluationDimension],
                     constraints: Optional[Dict[str, Any]] = None) -> DynamicTask:
        """Generate a dynamic evaluation task.
        
        Args:
            difficulty: Target difficulty level
            dimensions: Evaluation dimensions to focus on
            constraints: Additional constraints for task generation
            
        Returns:
            Generated dynamic task
        """
        constraints = constraints or {}
        
        # Select task type based on dimensions
        if EvaluationDimension.CORRECTNESS in dimensions:
            task_type = "code_explanation"
        elif EvaluationDimension.SECURITY in dimensions:
            task_type = "bug_detection" 
        elif EvaluationDimension.EFFICIENCY in dimensions:
            task_type = "optimization"
        else:
            task_type = random.choice(list(self.templates.keys()))
        
        # Generate unique task ID
        task_id = hashlib.sha256(f"{task_type}_{difficulty.name}_{time.time()}".encode()).hexdigest()[:16]
        
        # Get template
        template_data = random.choice(self.templates[task_type])
        template = template_data["template"]
        
        # Generate variables
        variables = {}
        for var in template_data["variables"]:
            if var == "language":
                variables[var] = constraints.get("language", "python")
            elif var == "code":
                variables[var] = self.generate_code_sample(difficulty, variables.get("language", "python"))
        
        # Fill template
        try:
            prompt = template.format(**variables)
        except KeyError as e:
            logger.error(f"Template formatting error: {e}")
            prompt = template  # Fallback to raw template
        
        return DynamicTask(
            task_id=task_id,
            prompt=prompt,
            expected_output=None,  # Would be generated by expert model or human annotation
            difficulty=difficulty,
            dimensions=dimensions,
            generation_params={
                "task_type": task_type,
                "variables": variables,
                "constraints": constraints
            }
        )


class CapabilityTracker:
    """Tracks model capabilities over time."""
    
    def __init__(self):
        """Initialize capability tracker."""
        self.capabilities: Dict[EvaluationDimension, ModelCapability] = {}
        self.history: List[EvaluationResult] = []
        self.adaptation_threshold = 0.1  # Score change threshold for adaptation
    
    def update_capability(self, dimension: EvaluationDimension, score: float) -> None:
        """Update capability for a dimension.
        
        Args:
            dimension: Evaluation dimension
            score: New score to incorporate
        """
        if dimension not in self.capabilities:
            self.capabilities[dimension] = ModelCapability(
                dimension=dimension,
                current_score=score,
                confidence_interval=(score - 0.1, score + 0.1),
                sample_size=1,
                last_updated=datetime.now()
            )
        else:
            cap = self.capabilities[dimension]
            
            # Update running average
            old_score = cap.current_score
            cap.sample_size += 1
            cap.current_score = (cap.current_score * (cap.sample_size - 1) + score) / cap.sample_size
            
            # Update confidence interval (simplified)
            recent_scores = [r.scores.get(dimension, 0) for r in self.history[-10:] if dimension in r.scores]
            if recent_scores:
                std = float(np.std(recent_scores))
                cap.confidence_interval = (
                    max(0.0, cap.current_score - 1.96 * std),
                    min(1.0, cap.current_score + 1.96 * std)
                )
            
            # Determine trend
            if cap.current_score > old_score + self.adaptation_threshold:
                cap.trend = "improving"
            elif cap.current_score < old_score - self.adaptation_threshold:
                cap.trend = "declining"
            else:
                cap.trend = "stable"
            
            cap.last_updated = datetime.now()
    
    def get_adaptive_difficulty(self, dimension: EvaluationDimension) -> DifficultyLevel:
        """Get adaptive difficulty for a dimension.
        
        Args:
            dimension: Evaluation dimension
            
        Returns:
            Recommended difficulty level
        """
        if dimension not in self.capabilities:
            return DifficultyLevel.MEDIUM  # Default
        
        cap = self.capabilities[dimension]
        score = cap.current_score
        
        # Adaptive difficulty based on current capability
        if score < 0.3:
            return DifficultyLevel.EASY
        elif score < 0.5:
            return DifficultyLevel.MEDIUM
        elif score < 0.7:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT
    
    def should_adapt_evaluation(self, dimension: EvaluationDimension) -> bool:
        """Check if evaluation should be adapted for a dimension.
        
        Args:
            dimension: Evaluation dimension
            
        Returns:
            True if adaptation is needed
        """
        if dimension not in self.capabilities:
            return True  # First evaluation
        
        cap = self.capabilities[dimension]
        
        # Adapt if trend is strong or confidence interval is wide
        ci_width = cap.confidence_interval[1] - cap.confidence_interval[0]
        return cap.trend != "stable" or ci_width > 0.3


class DynamicEvaluator:
    """Dynamic evaluation system that adapts to model capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dynamic evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.task_generator = TaskGenerator(self.config.get("template_path"))
        self.capability_tracker = CapabilityTracker()
        self.evaluation_history: List[EvaluationResult] = []
        self.min_sample_size = self.config.get("min_sample_size", 5)
        self.adaptation_frequency = self.config.get("adaptation_frequency", 10)
    
    async def evaluate_model(self, 
                           model_fn: Callable[[str], Any],
                           dimensions: List[EvaluationDimension],
                           num_tasks: int = 10) -> List[EvaluationResult]:
        """Evaluate model with dynamic task generation.
        
        Args:
            model_fn: Function that takes prompt and returns response
            dimensions: Evaluation dimensions to test
            num_tasks: Number of tasks to generate
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i in range(num_tasks):
            # Select dimension to focus on (round-robin or based on needs)
            focus_dimension = dimensions[i % len(dimensions)]
            
            # Get adaptive difficulty
            difficulty = self.capability_tracker.get_adaptive_difficulty(focus_dimension)
            
            # Generate task
            task = self.task_generator.generate_task(
                difficulty=difficulty,
                dimensions=[focus_dimension],
                constraints=self.config.get("task_constraints", {})
            )
            
            # Evaluate task
            start_time = time.time()
            try:
                model_response_raw = await asyncio.to_thread(model_fn, task.prompt)
                # Ensure we have a string response
                if asyncio.iscoroutine(model_response_raw):
                    model_response = await model_response_raw
                else:
                    model_response = str(model_response_raw)
                    
                execution_time = time.time() - start_time
                
                # Score response (simplified - would use sophisticated scoring)
                scores = await self._score_response(task, model_response)
                overall_score = sum(scores.values()) / len(scores)
                
                result = EvaluationResult(
                    task_id=task.task_id,
                    model_response=model_response,
                    scores=scores,
                    overall_score=overall_score,
                    execution_time=execution_time
                )
                
                results.append(result)
                self.evaluation_history.append(result)
                
                # Update capability tracking
                for dim, score in scores.items():
                    self.capability_tracker.update_capability(dim, score)
                
                logger.info(f"Task {i+1}/{num_tasks} completed - Score: {overall_score:.3f}")
                
            except Exception as e:
                logger.error(f"Evaluation failed for task {task.task_id}: {e}")
        
        return results
    
    async def _score_response(self, task: DynamicTask, response: str) -> Dict[EvaluationDimension, float]:
        """Score model response (simplified implementation).
        
        Args:
            task: Evaluation task
            response: Model response
            
        Returns:
            Scores for each dimension
        """
        scores = {}
        
        # Simplified scoring logic - in practice would use sophisticated metrics
        for dimension in task.dimensions:
            if dimension == EvaluationDimension.CORRECTNESS:
                # Basic correctness check (would use more sophisticated methods)
                score = 0.8 if len(response) > 50 and "error" not in response.lower() else 0.3
            elif dimension == EvaluationDimension.COMPLETENESS:
                # Check if response addresses the prompt comprehensively
                score = min(1.0, len(response) / 200)  # Simplified
            elif dimension == EvaluationDimension.CLARITY:
                # Basic clarity metrics
                sentences = response.split('.')
                avg_length = np.mean([len(s.split()) for s in sentences if s.strip()])
                score = max(0, 1 - (avg_length - 15) / 20)  # Penalize very long sentences
            else:
                # Default scoring
                score = random.uniform(0.4, 0.9)  # Placeholder
            
            scores[dimension] = max(0, min(1, score))
        
        return scores
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation results.
        
        Returns:
            Evaluation summary
        """
        if not self.evaluation_history:
            return {"message": "No evaluations completed"}
        
        # Calculate statistics
        all_scores = [r.overall_score for r in self.evaluation_history]
        execution_times = [r.execution_time for r in self.evaluation_history]
        
        # Capability breakdown
        capability_summary = {}
        for dimension, capability in self.capability_tracker.capabilities.items():
            capability_summary[dimension.value] = {
                "current_score": capability.current_score,
                "confidence_interval": capability.confidence_interval,
                "trend": capability.trend,
                "sample_size": capability.sample_size
            }
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "overall_statistics": {
                "mean_score": np.mean(all_scores),
                "std_score": np.std(all_scores),
                "min_score": np.min(all_scores),
                "max_score": np.max(all_scores),
                "mean_execution_time": np.mean(execution_times)
            },
            "capabilities": capability_summary,
            "recent_performance": {
                "last_10_average": np.mean(all_scores[-10:]) if len(all_scores) >= 10 else np.mean(all_scores),
                "trend": "improving" if len(all_scores) > 5 and np.mean(all_scores[-5:]) > np.mean(all_scores[:-5]) else "stable"
            }
        }
    
    def save_evaluation_state(self, filepath: str) -> None:
        """Save evaluation state to file.
        
        Args:
            filepath: Path to save state
        """
        try:
            state = {
                "config": self.config,
                "capabilities": {
                    dim.value: {
                        "current_score": cap.current_score,
                        "confidence_interval": cap.confidence_interval,
                        "sample_size": cap.sample_size,
                        "trend": cap.trend,
                        "last_updated": cap.last_updated.isoformat()
                    }
                    for dim, cap in self.capability_tracker.capabilities.items()
                },
                "evaluation_count": len(self.evaluation_history),
                "last_update": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Evaluation state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation state: {e}")


# Example usage
async def demo_dynamic_evaluation():
    """Demonstrate dynamic evaluation system."""
    
    # Mock model function
    def mock_model(prompt: str) -> str:
        time.sleep(0.1)  # Simulate processing time
        return f"Mock response to: {prompt[:50]}..."
    
    # Create evaluator
    evaluator = DynamicEvaluator({
        "min_sample_size": 3,
        "adaptation_frequency": 5,
        "task_constraints": {"language": "python"}
    })
    
    # Run evaluation
    dimensions = [
        EvaluationDimension.CORRECTNESS,
        EvaluationDimension.CLARITY,
        EvaluationDimension.COMPLETENESS
    ]
    
    results = await evaluator.evaluate_model(mock_model, dimensions, num_tasks=10)
    
    # Get summary
    summary = evaluator.get_evaluation_summary()
    print("Evaluation Summary:")
    print(json.dumps(summary, indent=2))
    
    return evaluator, results


if __name__ == "__main__":
    asyncio.run(demo_dynamic_evaluation())
