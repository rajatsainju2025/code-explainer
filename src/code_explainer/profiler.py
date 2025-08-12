"""Performance profiling and benchmarking utilities."""

import json
import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation: str
    duration_ms: float
    memory_peak_mb: float
    cpu_percent: float
    timestamp: float
    metadata: Dict[str, Any]


class PerformanceProfiler:
    """Profiles performance of code explanation operations."""

    def __init__(self):
        """Initialize the performance profiler."""
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()

    @contextmanager
    def profile(self, operation: str, **metadata):
        """Context manager for profiling operations.

        Args:
            operation: Name of the operation being profiled
            **metadata: Additional metadata to store
        """
        # Get initial measurements
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()

        # Reset CPU measurement for more accurate reading
        self.process.cpu_percent()

        try:
            yield
        finally:
            # Get final measurements
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self.process.cpu_percent()

            # Calculate metrics
            duration_ms = (end_time - start_time) * 1000
            memory_peak_mb = max(start_memory, end_memory)
            cpu_percent = max(start_cpu, end_cpu)

            # Store metrics
            metrics = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_peak_mb=memory_peak_mb,
                cpu_percent=cpu_percent,
                timestamp=start_time,
                metadata=metadata,
            )

            self.metrics.append(metrics)

            logger.debug(
                f"Operation '{operation}' completed in {duration_ms:.2f}ms "
                f"(Memory: {memory_peak_mb:.1f}MB, CPU: {cpu_percent:.1f}%)"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all recorded metrics."""
        if not self.metrics:
            return {"message": "No metrics recorded"}

        # Group by operation
        operations = {}
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)

        # Calculate statistics for each operation
        summary = {}
        for operation, metrics_list in operations.items():
            durations = [m.duration_ms for m in metrics_list]
            memories = [m.memory_peak_mb for m in metrics_list]

            summary[operation] = {
                "count": len(metrics_list),
                "duration_ms": {
                    "mean": statistics.mean(durations),
                    "median": statistics.median(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "stdev": statistics.stdev(durations) if len(durations) > 1 else 0,
                },
                "memory_mb": {
                    "mean": statistics.mean(memories),
                    "median": statistics.median(memories),
                    "min": min(memories),
                    "max": max(memories),
                    "stdev": statistics.stdev(memories) if len(memories) > 1 else 0,
                },
            }

        return summary

    def save_metrics(self, filepath: str) -> None:
        """Save metrics to a JSON file."""
        data = {
            "summary": self.get_summary(),
            "detailed_metrics": [
                {
                    "operation": m.operation,
                    "duration_ms": m.duration_ms,
                    "memory_peak_mb": m.memory_peak_mb,
                    "cpu_percent": m.cpu_percent,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.metrics
            ],
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Performance metrics saved to {filepath}")

    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()

    def benchmark_operation(
        self, operation_func: Callable, operation_name: str, num_iterations: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """Benchmark an operation multiple times.

        Args:
            operation_func: Function to benchmark
            operation_name: Name for the operation
            num_iterations: Number of times to run the operation
            **kwargs: Arguments to pass to the operation function

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking '{operation_name}' for {num_iterations} iterations...")

        results = []
        for i in range(num_iterations):
            with self.profile(f"{operation_name}_iteration_{i+1}"):
                try:
                    result = operation_func(**kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Iteration {i+1} failed: {e}")
                    results.append(None)

        # Get metrics for this benchmark
        benchmark_metrics = [
            m for m in self.metrics if m.operation.startswith(f"{operation_name}_iteration_")
        ]

        # Calculate benchmark summary
        if benchmark_metrics:
            durations = [m.duration_ms for m in benchmark_metrics]
            memories = [m.memory_peak_mb for m in benchmark_metrics]

            summary = {
                "operation": operation_name,
                "iterations": num_iterations,
                "successful_iterations": len([r for r in results if r is not None]),
                "duration_ms": {
                    "mean": statistics.mean(durations),
                    "median": statistics.median(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "stdev": statistics.stdev(durations) if len(durations) > 1 else 0,
                },
                "memory_mb": {
                    "mean": statistics.mean(memories),
                    "median": statistics.median(memories),
                    "min": min(memories),
                    "max": max(memories),
                    "stdev": statistics.stdev(memories) if len(memories) > 1 else 0,
                },
                "results": results,
            }
        else:
            summary = {
                "operation": operation_name,
                "iterations": num_iterations,
                "error": "No metrics recorded",
            }

        return summary


def benchmark_code_explainer(
    model_path: str = "./results",
    config_path: str = "configs/default.yaml",
    test_codes: Optional[List[str]] = None,
    num_iterations: int = 3,
) -> Dict[str, Any]:
    """Benchmark the code explainer with various operations.

    Args:
        model_path: Path to the trained model
        config_path: Path to configuration file
        test_codes: List of code snippets to test with
        num_iterations: Number of iterations for each test

    Returns:
        Dictionary with benchmark results
    """
    from .model import CodeExplainer

    if test_codes is None:
        test_codes = [
            "def hello(): print('hello world')",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def multiply(self, a, b):\n        return a * b",
            "import numpy as np\ndata = np.array([1, 2, 3, 4, 5])\nresult = np.mean(data)",
        ]

    profiler = PerformanceProfiler()

    # Benchmark model loading
    def load_model():
        return CodeExplainer(model_path=model_path, config_path=config_path)

    load_results = profiler.benchmark_operation(load_model, "model_loading", num_iterations=1)

    # Load the model once for explanation benchmarks
    explainer = CodeExplainer(model_path=model_path, config_path=config_path)

    # Benchmark different explanation strategies
    strategies = ["vanilla", "ast_augmented", "execution_trace"]
    strategy_results = {}

    for strategy in strategies:

        def explain_with_strategy(code, strat=strategy):
            return explainer.explain_code(code, strategy=strat)

        strategy_results[strategy] = {}
        for i, code in enumerate(test_codes):
            result = profiler.benchmark_operation(
                explain_with_strategy,
                f"explain_{strategy}_code_{i+1}",
                num_iterations=num_iterations,
                code=code,
            )
            strategy_results[strategy][f"code_{i+1}"] = result

    # Benchmark code analysis
    def analyze_code(code):
        return explainer.analyze_code(code)

    analysis_results = {}
    for i, code in enumerate(test_codes):
        result = profiler.benchmark_operation(
            analyze_code, f"analyze_code_{i+1}", num_iterations=num_iterations, code=code
        )
        analysis_results[f"code_{i+1}"] = result

    # Compile all results
    benchmark_results = {
        "model_loading": load_results,
        "explanation_strategies": strategy_results,
        "code_analysis": analysis_results,
        "overall_summary": profiler.get_summary(),
        "test_metadata": {
            "model_path": model_path,
            "config_path": config_path,
            "num_test_codes": len(test_codes),
            "num_iterations": num_iterations,
            "test_codes": test_codes,
        },
    }

    return benchmark_results


def main():
    """CLI entry point for performance benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark code explainer performance")
    parser.add_argument("--model-path", default="./results", help="Path to trained model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per test")
    parser.add_argument("--output", help="Output file for benchmark results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info("Starting performance benchmark...")

    try:
        results = benchmark_code_explainer(
            model_path=args.model_path, config_path=args.config, num_iterations=args.iterations
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))

        logger.info("Benchmark completed successfully")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
