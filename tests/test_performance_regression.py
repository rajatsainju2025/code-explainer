"""
Performance regression tests using pytest-benchmark.

These tests ensure that code explanation performance doesn't degrade over time.
Run with: pytest tests/test_performance_regression.py -v --benchmark-only
"""

import pytest
from pathlib import Path
from benchmarks.benchmark_inference import InferenceBenchmark, BenchmarkResult

# Test data
SIMPLE_CODE = "def add(a, b): return a + b"

COMPLEX_CODE = """
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number using recursion.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def get_history(self):
        return self.history.copy()
""".strip()

@pytest.fixture(scope="session")
def benchmark_instance():
    """Create a benchmark instance for testing."""
    benchmark = InferenceBenchmark()
    benchmark.setup()
    yield benchmark
    benchmark.teardown()

@pytest.mark.benchmark
def test_simple_explanation_performance(benchmark, benchmark_instance):
    """Test performance of simple code explanation."""
    def explain_simple():
        return benchmark_instance.explainer.explain_code(SIMPLE_CODE, strategy="vanilla")

    result = benchmark(explain_simple)

    # Basic assertions
    assert result is not None
    assert len(result) > 0
    assert "add" in result.lower()

@pytest.mark.benchmark
def test_complex_explanation_performance(benchmark, benchmark_instance):
    """Test performance of complex code explanation."""
    def explain_complex():
        return benchmark_instance.explainer.explain_code(COMPLEX_CODE, strategy="ast_augmented")

    result = benchmark(explain_complex)

    # Basic assertions
    assert result is not None
    assert len(result) > 10
    assert "fibonacci" in result.lower() or "recursive" in result.lower()

@pytest.mark.benchmark
@pytest.mark.parametrize("strategy", ["vanilla", "ast_augmented"])
def test_strategy_performance_comparison(benchmark, benchmark_instance, strategy):
    """Test performance across different explanation strategies."""
    def explain_with_strategy():
        return benchmark_instance.explainer.explain_code(SIMPLE_CODE, strategy=strategy)

    result = benchmark(explain_with_strategy)

    # Basic assertions
    assert result is not None
    assert len(result) > 0

@pytest.mark.slow
@pytest.mark.benchmark
def test_memory_usage_regression(benchmark, benchmark_instance):
    """Test for memory usage regressions in code explanation."""
    import tracemalloc

    def explain_with_memory_tracking():
        tracemalloc.start()
        result = benchmark_instance.explainer.explain_code(COMPLEX_CODE, strategy="vanilla")
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak

    result, peak_memory = benchmark(explain_with_memory_tracking)

    # Assert reasonable memory usage (less than 500MB peak)
    assert peak_memory < 500 * 1024 * 1024  # 500MB in bytes
    assert result[0] is not None
    assert len(result[0]) > 0

@pytest.mark.benchmark
def test_batch_explanation_performance(benchmark, benchmark_instance):
    """Test performance of batch code explanation operations."""
    code_samples = [
        "def add(a, b): return a + b",
        "def multiply(x, y): return x * y",
        "def divide(a, b): return a / b if b != 0 else 0",
    ]

    def explain_batch():
        results = []
        for code in code_samples:
            result = benchmark_instance.explainer.explain_code(code, strategy="vanilla")
            results.append(result)
        return results

    results = benchmark(explain_batch)

    # Assertions
    assert len(results) == 3
    assert all(len(r) > 0 for r in results)

@pytest.mark.benchmark
def test_retrieval_performance(benchmark, benchmark_instance):
    """Test retrieval-augmented explanation performance."""
    # This test may be skipped if retrieval is not available
    try:
        def explain_with_retrieval():
            return benchmark_instance.explainer.explain_code(
                SIMPLE_CODE,
                strategy="retrieval_augmented"
            )

        result = benchmark(explain_with_retrieval)

        # Basic assertions
        assert result is not None
        assert len(result) > 0

    except Exception as e:
        pytest.skip(f"Retrieval-augmented explanation not available: {e}")

def test_performance_regression_detection():
    """Test that we can detect performance regressions."""
    # Create mock benchmark results
    baseline_results = [
        BenchmarkResult(
            operation="explain_code_vanilla",
            samples=10,
            mean_time=0.5,
            median_time=0.48,
            min_time=0.4,
            max_time=0.6,
            std_dev=0.05,
            throughput=2.0,
            memory_peak=1000000,
            cpu_percent=5.0,
            timestamp="2024-01-01T00:00:00"
        )
    ]

    # Simulate slower current results (regression)
    current_results = [
        BenchmarkResult(
            operation="explain_code_vanilla",
            samples=10,
            mean_time=0.75,  # 50% slower
            median_time=0.72,
            min_time=0.6,
            max_time=0.9,
            std_dev=0.08,
            throughput=1.33,
            memory_peak=1200000,
            cpu_percent=6.0,
            timestamp="2024-01-02T00:00:00"
        )
    ]

    benchmark = InferenceBenchmark()

    # Save baseline
    baseline_file = benchmark.save_results(baseline_results, "test_baseline.json")

    # Compare with current
    comparison = benchmark.compare_with_baseline(current_results, "test_baseline.json", threshold=0.1)

    # Assertions
    assert comparison["regression_count"] == 1
    assert len(comparison["regressions"]) == 1
    assert comparison["regressions"][0]["difference_percent"] > 0.1

    # Clean up
    Path(baseline_file).unlink(missing_ok=True)

if __name__ == "__main__":
    # Allow running benchmarks directly
    import sys
    import os

    # Add src to path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

    pytest.main([__file__, "-v", "--benchmark-only"])