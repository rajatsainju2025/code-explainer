#!/usr/bin/env python3
"""
Performance Benchmarking Script for Code Explainer

This script provides comprehensive performance benchmarking capabilities
for the Code Explainer system, measuring throughput, latency, memory usage,
and other key performance metrics.

Usage:
    python benchmark_performance.py --model-path ./results --config configs/default.yaml
    python benchmark_performance.py --compare-strategies --output benchmark_results.json
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_explainer.model import CodeExplainer
from code_explainer.performance_optimization import PerformanceOptimizer
from code_explainer.profiler import PerformanceProfiler


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self, model_path: str, config_path: str):
        """Initialize benchmark suite."""
        self.model_path = model_path
        self.config_path = config_path
        self.explainer = None
        self.profiler = PerformanceProfiler()
        self.optimizer = PerformanceOptimizer()

        # Benchmark test cases
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load benchmark test cases."""
        return [
            {
                "name": "simple_function",
                "code": "def hello(name): return f'Hello, {name}!'",
                "description": "Simple function with string formatting"
            },
            {
                "name": "fibonacci",
                "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "description": "Recursive fibonacci implementation"
            },
            {
                "name": "class_example",
                "code": "class Calculator:\n    def __init__(self):\n        self.value = 0\n\n    def add(self, x):\n        self.value += x\n        return self.value",
                "description": "Simple class with methods"
            },
            {
                "name": "list_comprehension",
                "code": "def process_list(items):\n    return [x**2 for x in items if x > 0]",
                "description": "List comprehension with filtering"
            },
            {
                "name": "complex_algorithm",
                "code": """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)""",
                "description": "Quicksort algorithm implementation"
            }
        ]

    def setup(self) -> None:
        """Setup benchmark environment."""
        print("Setting up benchmark environment...")
        self.explainer = CodeExplainer(
            model_path=self.model_path,
            config_path=self.config_path
        )
        print("✓ Model loaded successfully")

    def _ensure_explainer(self) -> None:
        """Ensure explainer is initialized."""
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call setup() first.")

    def benchmark_single_explanation(self, code: str, strategy: str = "vanilla") -> Dict[str, Any]:
        """Benchmark a single code explanation."""
        self._ensure_explainer()
        assert self.explainer is not None  # Type assertion for linter

        with self.profiler.profile("single_explanation"):
            explanation = self.explainer.explain_code(code, strategy=strategy)

        return {
            "code_length": len(code),
            "explanation_length": len(explanation),
            "strategy": strategy
        }

    def benchmark_batch_explanation(self, codes: List[str], batch_size: int = 10) -> Dict[str, Any]:
        """Benchmark batch code explanation."""
        self._ensure_explainer()
        assert self.explainer is not None  # Type assertion for linter

        with self.profiler.profile("batch_explanation"):
            explanations = self.explainer.explain_code_batch(codes)

        return {
            "batch_size": len(codes),
            "total_explanations": len(explanations),
            "avg_explanation_length": statistics.mean(len(exp) for exp in explanations)
        }

    def benchmark_parallel_explanation(self, codes: List[str], max_workers: int = 4) -> Dict[str, Any]:
        """Benchmark parallel code explanation."""
        self._ensure_explainer()
        assert self.explainer is not None  # Type assertion for linter

        with self.profiler.profile("parallel_explanation"):
            explanations = self.explainer.explain_code_parallel(codes, max_workers=max_workers)

        return {
            "batch_size": len(codes),
            "max_workers": max_workers,
            "total_explanations": len(explanations)
        }

    def compare_strategies(self) -> Dict[str, Any]:
        """Compare different explanation strategies."""
        self._ensure_explainer()
        assert self.explainer is not None  # Type assertion for linter

        strategies = ["vanilla", "ast_augmented", "execution_trace"]
        results = {}

        test_code = self.test_cases[1]["code"]  # Use fibonacci as test case

        for strategy in strategies:
            try:
                with self.profiler.profile(f"strategy_{strategy}"):
                    explanation = self.explainer.explain_code(test_code, strategy=strategy)

                results[strategy] = {
                    "success": True,
                    "explanation_length": len(explanation),
                    "strategy": strategy
                }
            except Exception as e:
                results[strategy] = {
                    "success": False,
                    "error": str(e),
                    "strategy": strategy
                }

        return results

    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        self._ensure_explainer()
        assert self.explainer is not None  # Type assertion for linter

        import psutil
        process = psutil.Process()

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run several explanations
        codes = [case["code"] for case in self.test_cases[:3]]

        with self.profiler.profile("memory_test"):
            explanations = self.explainer.explain_code_batch(codes)

        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "memory_delta_mb": peak_memory - baseline_memory,
            "test_cases": len(codes)
        }

    def benchmark_throughput(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Benchmark throughput over a time period."""
        self._ensure_explainer()
        assert self.explainer is not None  # Type assertion for linter

        test_code = self.test_cases[0]["code"]  # Simple test case
        start_time = time.time()
        end_time = start_time + duration_seconds

        explanations_count = 0
        total_chars = 0

        while time.time() < end_time:
            explanation = self.explainer.explain_code(test_code)
            explanations_count += 1
            total_chars += len(explanation)

        actual_duration = time.time() - start_time

        return {
            "duration_seconds": actual_duration,
            "explanations_count": explanations_count,
            "throughput_per_second": explanations_count / actual_duration,
            "avg_chars_per_explanation": total_chars / explanations_count,
            "total_chars": total_chars
        }

    def benchmark_strategies_comparison(self) -> Dict[str, Any]:
        """Compare different explanation strategies."""
        self._ensure_explainer()
        assert self.explainer is not None  # Type assertion for linter

        strategies = ["vanilla", "ast_augmented", "execution_trace"]
        results = {}

        test_code = self.test_cases[1]["code"]  # Use fibonacci as test case

        for strategy in strategies:
            try:
                with self.profiler.profile(f"strategy_{strategy}"):
                    explanation = self.explainer.explain_code(test_code, strategy=strategy)

                results[strategy] = {
                    "success": True,
                    "explanation_length": len(explanation),
                    "strategy": strategy
                }
            except Exception as e:
                results[strategy] = {
                    "success": False,
                    "error": str(e),
                    "strategy": strategy
                }

        return results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("Running full performance benchmark...")

        results = {
            "timestamp": time.time(),
            "model_path": self.model_path,
            "config_path": self.config_path,
            "test_cases": len(self.test_cases)
        }

        # Single explanation benchmark
        print("  • Benchmarking single explanations...")
        single_results = []
        for case in self.test_cases:
            result = self.benchmark_single_explanation(case["code"])
            result["test_case"] = case["name"]
            single_results.append(result)
        results["single_explanation"] = single_results

        # Batch explanation benchmark
        print("  • Benchmarking batch explanations...")
        codes = [case["code"] for case in self.test_cases]
        results["batch_explanation"] = self.benchmark_batch_explanation(codes)

        # Parallel explanation benchmark
        print("  • Benchmarking parallel explanations...")
        results["parallel_explanation"] = self.benchmark_parallel_explanation(codes)

        # Strategy comparison
        print("  • Comparing explanation strategies...")
        results["strategy_comparison"] = self.benchmark_strategies_comparison()

        # Memory usage
        print("  • Measuring memory usage...")
        results["memory_usage"] = self.benchmark_memory_usage()

        # Throughput
        print("  • Measuring throughput...")
        results["throughput"] = self.benchmark_throughput(duration_seconds=10)

        # Overall profiler summary
        results["profiler_summary"] = self.profiler.get_summary()

        print("✓ Benchmark completed")
        return results

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save benchmark results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✓ Results saved to {output_path}")

        # Also save profiler metrics
        profiler_file = output_file.parent / f"{output_file.stem}_profiler.json"
        self.profiler.save_metrics(str(profiler_file))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Performance Benchmarking for Code Explainer")
    parser.add_argument("--model-path", default="./results", help="Path to trained model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--compare-strategies", action="store_true", help="Compare different strategies")
    parser.add_argument("--throughput-test", type=int, help="Run throughput test for N seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Initialize benchmark
    benchmark = PerformanceBenchmark(args.model_path, args.config)

    try:
        benchmark.setup()

        if args.compare_strategies:
            print("Comparing explanation strategies...")
            results = {"strategy_comparison": benchmark.benchmark_strategies_comparison()}
        elif args.throughput_test:
            print(f"Running throughput test for {args.throughput_test} seconds...")
            results = {"throughput": benchmark.benchmark_throughput(args.throughput_test)}
        else:
            results = benchmark.run_full_benchmark()

        # Output results
        if args.output:
            benchmark.save_results(results, args.output)
        else:
            print(json.dumps(results, indent=2, default=str))

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise


if __name__ == "__main__":
    main()