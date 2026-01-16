"""
Comprehensive Inference Benchmarking Suite

This module provides detailed performance benchmarking for code explanation inference,
including regression testing, statistical analysis, and automated performance monitoring.
"""

import time
import statistics
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import psutil
import tracemalloc
from functools import lru_cache

from code_explainer import CodeExplainer

@lru_cache(maxsize=4)
def _get_cached_explainer(model_path: Optional[str], config_path: Optional[str]) -> CodeExplainer:
    """Cache CodeExplainer instances to avoid repeated initialization."""
    return CodeExplainer(model_path=model_path, config_path=config_path)

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    samples: int
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float  # operations per second
    memory_peak: int   # peak memory usage in bytes
    cpu_percent: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "samples": self.samples,
            "mean_time": self.mean_time,
            "median_time": self.median_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "std_dev": self.std_dev,
            "throughput": self.throughput,
            "memory_peak": self.memory_peak,
            "cpu_percent": self.cpu_percent,
            "timestamp": self.timestamp,
            **self.metadata
        }

class InferenceBenchmark:
    """Comprehensive inference benchmarking suite."""    
    __slots__ = ('model_path', 'config_path', 'explainer', 'results_dir')
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config_path = config_path
        self.explainer = None
        self.results_dir = Path("benchmarks/results")
        self.results_dir.mkdir(exist_ok=True)

    def setup(self):
        """Initialize the code explainer."""
        if self.explainer is None:
            logger.info("Initializing CodeExplainer for benchmarking...")
            self.explainer = _get_cached_explainer(self.model_path, self.config_path)

    def teardown(self):
        """Clean up resources."""
        if self.explainer:
            # Clean up any resources if needed
            pass

    def benchmark_explanation(
        self,
        code: str,
        strategy: str = "vanilla",
        num_samples: int = 10,
        warmup_samples: int = 2
    ) -> BenchmarkResult:
        """Benchmark code explanation performance."""
        if self.explainer is None:
            raise RuntimeError("Benchmark not properly initialized. Call setup() first.")

        # Warmup runs
        logger.info("Running %d warmup samples...", warmup_samples)
        for _ in range(warmup_samples):
            _ = self.explainer.explain_code(code, strategy=strategy)

        # Benchmark runs with optimized memory tracing
        logger.info("Running %d benchmark samples...", num_samples)
        times = []
        memory_peaks = []

        process = psutil.Process()
        initial_cpu = process.cpu_percent()

        # Start memory tracing once for all samples
        tracemalloc.start()
        initial_memory = process.memory_info().rss

        result = ""  # Initialize result variable
        for i in range(num_samples):
            start_time = time.perf_counter()
            result = self.explainer.explain_code(code, strategy=strategy)
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            times.append(elapsed)
            logger.debug("Sample %d: %.3fs", i, elapsed)

        # Get final memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peaks.append(peak)

        final_cpu = process.cpu_percent()

        # Calculate statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = 1.0 / mean_time if mean_time > 0 else 0.0
        memory_peak = memory_peaks[0] if memory_peaks else 0
        cpu_percent = final_cpu - initial_cpu

        result = BenchmarkResult(
            operation=f"explain_code_{strategy}",
            samples=num_samples,
            mean_time=mean_time,
            median_time=median_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_peak=memory_peak,
            cpu_percent=cpu_percent,
            timestamp=datetime.now().isoformat(),
            metadata={
                "code_length": len(code),
                "strategy": strategy,
                "result_length": len(result) if result else 0,
                "model_path": self.model_path,
                "config_path": self.config_path
            }
        )

        logger.info("Benchmark completed: %.3fs mean, %.2f ops/sec", mean_time, throughput)
        return result

    def benchmark_multiple_strategies(
        self,
        code: str,
        strategies: Optional[List[str]] = None,
        num_samples: int = 5
    ) -> List[BenchmarkResult]:
        """Benchmark multiple explanation strategies."""
        if strategies is None:
            strategies = ["vanilla", "ast_augmented", "retrieval_augmented", "execution_trace"]

        results = []
        for strategy in strategies:
            try:
                logger.info("Benchmarking strategy: %s", strategy)
                result = self.benchmark_explanation(code, strategy, num_samples)
                results.append(result)
            except (ImportError, RuntimeError, ValueError, TypeError) as e:
                logger.error("Failed to benchmark strategy %s: %s", strategy, e)
                # Create error result
                results.append(BenchmarkResult(
                    operation=f"explain_code_{strategy}",
                    samples=0,
                    mean_time=0.0,
                    median_time=0.0,
                    min_time=0.0,
                    max_time=0.0,
                    std_dev=0.0,
                    throughput=0.0,
                    memory_peak=0,
                    cpu_percent=0.0,
                    timestamp=datetime.now().isoformat(),
                    metadata={"error": str(e), "strategy": strategy}
                ))

        return results

    def save_results(self, results: List[BenchmarkResult], filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        filepath = self.results_dir / filename
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                "python_version": f"{psutil.version_info}",
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total
            },
            "results": [r.to_dict() for r in results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info("Results saved to %s", filepath)
        return filepath

    def load_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """Load baseline results for comparison."""
        filepath = self.results_dir / baseline_file
        if not filepath.exists():
            raise FileNotFoundError(f"Baseline file not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def compare_with_baseline(
        self,
        current_results: List[BenchmarkResult],
        baseline_file: str,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Compare current results with baseline and detect regressions."""
        baseline_data = self.load_baseline(baseline_file)
        baseline_results = {r["operation"]: r for r in baseline_data["results"]}

        comparisons = {}
        regressions = []

        for result in current_results:
            baseline = baseline_results.get(result.operation)
            if baseline:
                current_mean = result.mean_time
                baseline_mean = baseline["mean_time"]
                diff = current_mean - baseline_mean
                diff_percent = diff / baseline_mean if baseline_mean > 0 else 0

                comparison = {
                    "operation": result.operation,
                    "current_mean": current_mean,
                    "baseline_mean": baseline_mean,
                    "difference": diff,
                    "difference_percent": diff_percent,
                    "regression": diff_percent > threshold
                }
                comparisons[result.operation] = comparison

                if comparison["regression"]:
                    regressions.append(comparison)
            else:
                logger.warning("No baseline found for operation: %s", result.operation)

        return {
            "comparisons": comparisons,
            "regressions": regressions,
            "regression_count": len(regressions),
            "threshold": threshold
        }

    def load_test_cases_from_file(self, filepath: str) -> List[tuple[str, str]]:
        """Load test cases from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return [(case['name'], case['code']) for case in data.get('test_cases', [])]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load test cases from %s: %s", filepath, e)
            return []

def run_comprehensive_benchmark(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    output_file: Optional[str] = None,
    compare_baseline: Optional[str] = None,
    test_cases_file: Optional[str] = None,
    num_samples: int = 3
):
    """Run comprehensive benchmarking suite."""
    benchmark = InferenceBenchmark(model_path, config_path)

    # Load test cases
    if test_cases_file:
        test_cases = benchmark.load_test_cases_from_file(test_cases_file)
        if not test_cases:
            logger.warning("No test cases loaded from %s, using defaults", test_cases_file)
            test_cases = get_default_test_cases()
    else:
        test_cases = get_default_test_cases()

    all_results = []

    for name, code in test_cases:
        logger.info("Benchmarking: %s", name)
        results = benchmark.benchmark_multiple_strategies(code, num_samples=num_samples)
        all_results.extend(results)

    # Save results
    if output_file:
        filepath = benchmark.save_results(all_results, output_file)
    else:
        filepath = benchmark.save_results(all_results)

    # Compare with baseline if requested
    if compare_baseline:
        comparison = benchmark.compare_with_baseline(all_results, compare_baseline)
        comparison_file = filepath.with_suffix('.comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info("Comparison saved to %s", comparison_file)

        if comparison["regression_count"] > 0:
            logger.warning("⚠️  Performance regressions detected: %d", comparison['regression_count'])
            for reg in comparison["regressions"]:
                logger.warning(".1f")
        else:
            logger.info("✅ No performance regressions detected")

    benchmark.teardown()
    return all_results


def get_default_test_cases() -> List[tuple[str, str]]:
    """Get default test cases for benchmarking."""
    return [
        ("Simple function", "def add(a, b): return a + b"),
        ("Complex algorithm", """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""".strip()),
        ("Class with methods", """
class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def get_history(self):
        return self.history
""".strip())
    ]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Explainer Inference Benchmark")
    parser.add_argument("--model-path", help="Path to model directory")
    parser.add_argument("--config-path", help="Path to config file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--baseline", help="Baseline file for comparison")
    parser.add_argument("--test-cases", help="JSON file containing test cases")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per benchmark")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = run_comprehensive_benchmark(
        model_path=args.model_path,
        config_path=args.config_path,
        output_file=args.output,
        compare_baseline=args.baseline,
        test_cases_file=args.test_cases,
        num_samples=args.samples
    )

    print(f"\nBenchmark completed with {len(results)} results")
    for result in results:
        if result.samples > 0:  # Skip error results
            print(".3f")
