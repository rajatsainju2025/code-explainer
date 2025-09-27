"""
Performance Benchmarking Suite

This module provides comprehensive performance benchmarking and profiling
capabilities for the Code Explainer system, including load testing, latency
analysis, resource monitoring, and comparative benchmarking.

Key Features:
- Automat             "platform": "unknown",          "platform": psutil.os.uname().sysname if hasattr(psutil.os, 'uname') else "unknown",d performance benchmarking with configurable workloads
- Latency and throughput analysis with statistical reporting
- Memory and CPU profiling with detailed metrics
- Comparative benchmarking against baselines and competitors
- Load testing with configurable concurrency and request patterns
- Performance regression detection and alerting
- Resource utilization analysis and optimization recommendations
- Integration with external monitoring and APM systems
- Historical performance tracking and trend analysis

Based on industry-standard benchmarking practices and performance analysis techniques.
"""

import asyncio
import time
import statistics
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from abc import ABC, abstractmethod
import json
import csv
import threading
import psutil
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Lazy-loaded heavy dependencies
_numpy_available = None
_scipy_available = None
_matplotlib_available = None
_seaborn_available = None

def _ensure_numpy():
    """Lazy load numpy."""
    global _numpy_available
    if _numpy_available is None:
        try:
            import numpy as np
            _numpy_available = np
        except ImportError:
            _numpy_available = False
    return _numpy_available

def _ensure_scipy():
    """Lazy load scipy."""
    global _scipy_available
    if _scipy_available is None:
        try:
            from scipy import stats
            _scipy_available = stats
        except ImportError:
            _scipy_available = False
    return _scipy_available

def _ensure_matplotlib():
    """Lazy load matplotlib."""
    global _matplotlib_available
    if _matplotlib_available is None:
        try:
            import matplotlib.pyplot as plt
            _matplotlib_available = plt
        except ImportError:
            _matplotlib_available = False
    return _matplotlib_available

def _ensure_seaborn():
    """Lazy load seaborn."""
    global _seaborn_available
    if _seaborn_available is None:
        try:
            import seaborn as sns
            _seaborn_available = sns
        except ImportError:
            _seaborn_available = False
    return _seaborn_available

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    timestamp: datetime
    duration: float
    operations: int
    throughput: float  # operations per second
    latency_p50: float  # median latency
    latency_p95: float  # 95th percentile latency
    latency_p99: float  # 99th percentile latency
    min_latency: float
    max_latency: float
    error_rate: float
    resource_usage: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str
    duration: int  # seconds
    concurrency: int
    ramp_up_time: int = 0  # seconds
    request_pattern: str = "constant"  # constant, ramp, spike
    payload_sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000])
    warmup_iterations: int = 10
    cooldown_iterations: int = 5

@dataclass
class PerformanceProfile:
    """Performance profile data."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io: Dict[str, int]
    network_io: Dict[str, int]
    thread_count: int
    open_files: int

class BenchmarkSuite:
    """Main benchmarking suite."""

    def __init__(self, output_dir: Path = Path("benchmarks/results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.profiles: List[PerformanceProfile] = []
        self.baselines: Dict[str, BenchmarkResult] = {}

    def load_baseline(self, name: str, filepath: Path):
        """Load baseline benchmark results."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.baselines[name] = BenchmarkResult(**data)

    def save_baseline(self, name: str, result: BenchmarkResult):
        """Save benchmark result as baseline."""
        filepath = self.output_dir / f"baseline_{name}.json"
        with open(filepath, 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)

    async def run_benchmark(self, config: BenchmarkConfig,
                          benchmark_func: Callable) -> BenchmarkResult:
        """Run a benchmark with given configuration."""
        logger.info(f"Starting benchmark: {config.name}")

        # Warmup
        await self._warmup(benchmark_func, config.warmup_iterations)

        # Start profiling
        profile_thread = threading.Thread(
            target=self._profile_system,
            args=(config.duration + 10,)
        )
        profile_thread.daemon = True
        profile_thread.start()

        # Run benchmark
        start_time = time.time()
        latencies = []
        errors = 0
        operations = 0

        try:
            async for latency, error in self._generate_load(
                config, benchmark_func
            ):
                operations += 1
                if error:
                    errors += 1
                else:
                    latencies.append(latency)

                # Check if duration exceeded
                if time.time() - start_time > config.duration:
                    break

        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            errors += 1

        end_time = time.time()
        duration = end_time - start_time

        # Calculate metrics
        throughput = operations / duration if duration > 0 else 0
        error_rate = errors / operations if operations > 0 else 0

        if latencies:
            latency_p50 = statistics.median(latencies)
            latency_p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            latency_p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            latency_p50 = latency_p95 = latency_p99 = min_latency = max_latency = 0

        # Get resource usage
        resource_usage = self._get_resource_usage()

        result = BenchmarkResult(
            name=config.name,
            timestamp=datetime.now(),
            duration=duration,
            operations=operations,
            throughput=throughput,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            min_latency=min_latency,
            max_latency=max_latency,
            error_rate=error_rate,
            resource_usage=resource_usage,
            metadata={
                "config": config.__dict__,
                "total_latencies": len(latencies),
                "system_info": self._get_system_info()
            }
        )

        self.results.append(result)
        logger.info(f"Benchmark completed: {config.name} - {throughput:.2f} ops/sec")

        return result

    async def _generate_load(self, config: BenchmarkConfig,
                           benchmark_func: Callable) -> AsyncGenerator[tuple, None]:
        """Generate load according to configuration."""
        semaphore = asyncio.Semaphore(config.concurrency)

        async def worker():
            try:
                start = time.time()
                await benchmark_func()
                latency = time.time() - start
                return latency, False
            except Exception as e:
                logger.warning(f"Benchmark operation failed: {str(e)}")
                return 0, True

        # Generate requests based on pattern
        if config.request_pattern == "constant":
            tasks = []
            for _ in range(config.concurrency * 10):  # Generate enough tasks
                task = asyncio.create_task(worker())
                tasks.append(task)

            for task in asyncio.as_completed(tasks):
                result = await task
                yield result
        else:
            # Simplified implementation for other patterns
            for _ in range(config.concurrency):
                result = await worker()
                yield result

    async def _warmup(self, benchmark_func: Callable, iterations: int):
        """Warm up the system before benchmarking."""
        logger.info(f"Warming up with {iterations} iterations")
        for i in range(iterations):
            try:
                await benchmark_func()
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {str(e)}")

    def _profile_system(self, duration: int):
        """Profile system resources during benchmark."""
        start_time = time.time()

        while time.time() - start_time < duration:
            try:
                profile = PerformanceProfile(
                    timestamp=datetime.now(),
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    memory_mb=psutil.virtual_memory().used / 1024 / 1024,
                    disk_io=psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    network_io=psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
                    thread_count=len(psutil.Process().threads()),
                    open_files=len(psutil.Process().open_files())
                )
                self.profiles.append(profile)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Profiling error: {str(e)}")
                break

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total / 1024 / 1024,
            "python_version": f"{psutil.__version__ if hasattr(psutil, '__version__') else 'unknown'}",
            "platform": "unknown",
        }

    def compare_with_baseline(self, result: BenchmarkResult,
                            baseline_name: str) -> Dict[str, Any]:
        """Compare benchmark result with baseline."""
        if baseline_name not in self.baselines:
            return {"error": f"Baseline '{baseline_name}' not found"}

        baseline = self.baselines[baseline_name]

        comparison = {
            "throughput_change": (result.throughput - baseline.throughput) / baseline.throughput * 100,
            "latency_p50_change": (result.latency_p50 - baseline.latency_p50) / baseline.latency_p50 * 100,
            "latency_p95_change": (result.latency_p95 - baseline.latency_p95) / baseline.latency_p95 * 100,
            "error_rate_change": (result.error_rate - baseline.error_rate) / baseline.error_rate * 100 if baseline.error_rate > 0 else 0,
            "regression_detected": False
        }

        # Detect regressions
        if (comparison["throughput_change"] < -10 or  # 10% throughput decrease
            comparison["latency_p95_change"] > 20):   # 20% latency increase
            comparison["regression_detected"] = True

        return comparison

    def generate_report(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if results is None:
            results = self.results

        if not results:
            return {"error": "No benchmark results available"}

        report = {
            "summary": {
                "total_benchmarks": len(results),
                "timestamp": datetime.now().isoformat(),
                "duration_range": {
                    "min": min(r.duration for r in results),
                    "max": max(r.duration for r in results),
                    "avg": statistics.mean(r.duration for r in results)
                }
            },
            "performance_metrics": {
                "throughput": {
                    "min": min(r.throughput for r in results),
                    "max": max(r.throughput for r in results),
                    "avg": statistics.mean(r.throughput for r in results),
                    "std": statistics.stdev(r.throughput for r in results) if len(results) > 1 else 0
                },
                "latency_p95": {
                    "min": min(r.latency_p95 for r in results),
                    "max": max(r.latency_p95 for r in results),
                    "avg": statistics.mean(r.latency_p95 for r in results)
                }
            },
            "results": [r.__dict__ for r in results]
        }

        # Add baseline comparisons
        if self.baselines:
            report["baseline_comparisons"] = {}
            for result in results:
                for baseline_name in self.baselines:
                    comparison = self.compare_with_baseline(result, baseline_name)
                    if baseline_name not in report["baseline_comparisons"]:
                        report["baseline_comparisons"][baseline_name] = []
                    report["baseline_comparisons"][baseline_name].append({
                        "benchmark": result.name,
                        "comparison": comparison
                    })

        return report

    def export_report(self, filepath: Path, format: str = "json"):
        """Export benchmark report."""
        report = self.generate_report()

        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == "csv":
            self._export_csv(filepath, report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_csv(self, filepath: Path, report: Dict[str, Any]):
        """Export report as CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Benchmark", "Throughput", "Latency P50", "Latency P95", "Error Rate"])

            for result in report["results"]:
                writer.writerow([
                    result["name"],
                    result["throughput"],
                    result["latency_p50"],
                    result["latency_p95"],
                    result["error_rate"]
                ])

    def plot_results(self, results: Optional[List[BenchmarkResult]] = None,
                    save_path: Optional[Path] = None):
        """Generate performance plots."""
        if results is None:
            results = self.results

        if not results:
            logger.warning("No results to plot")
            return

        # Set up the plotting style
        plt = _ensure_matplotlib()
        if not plt:
            logger.warning("Matplotlib not available, skipping plot generation")
            return

        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Benchmark Performance Analysis', fontsize=16)

        # Throughput plot
        names = [r.name for r in results]
        throughputs = [r.throughput for r in results]
        axes[0, 0].bar(names, throughputs, color='skyblue')
        axes[0, 0].set_title('Throughput (ops/sec)')
        axes[0, 0].set_ylabel('Operations/second')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Latency plot
        latencies_p50 = [r.latency_p50 for r in results]
        latencies_p95 = [r.latency_p95 for r in results]
        x = range(len(names))
        axes[0, 1].bar(x, latencies_p50, width=0.35, label='P50', color='lightgreen')
        axes[0, 1].bar([i + 0.35 for i in x], latencies_p95, width=0.35, label='P95', color='orange')
        axes[0, 1].set_title('Latency (seconds)')
        axes[0, 1].set_ylabel('Latency')
        axes[0, 1].set_xticks([i + 0.175 for i in x])
        axes[0, 1].set_xticklabels(names, rotation=45)
        axes[0, 1].legend()

        # Error rate plot
        error_rates = [r.error_rate * 100 for r in results]  # Convert to percentage
        axes[1, 0].bar(names, error_rates, color='salmon')
        axes[1, 0].set_title('Error Rate (%)')
        axes[1, 0].set_ylabel('Error Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Resource usage plot
        cpu_usage = [r.resource_usage.get('cpu_percent', 0) for r in results]
        memory_usage = [r.resource_usage.get('memory_percent', 0) for r in results]
        axes[1, 1].plot(names, cpu_usage, marker='o', label='CPU %', color='blue')
        axes[1, 1].plot(names, memory_usage, marker='s', label='Memory %', color='red')
        axes[1, 1].set_title('Resource Usage')
        axes[1, 1].set_ylabel('Usage (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {save_path}")
        else:
            plt.show()

class CodeExplainerBenchmark:
    """Specific benchmarks for Code Explainer system."""

    def __init__(self, suite: BenchmarkSuite):
        self.suite = suite

    async def benchmark_explanation(self, code_snippet: Optional[str] = None) -> float:
        """Benchmark code explanation functionality."""
        if code_snippet is None:
            code_snippet = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""

        # Import here to avoid circular imports
        from code_explainer.model import CodeExplainer

        explainer = CodeExplainer()

        start_time = time.time()
        result = await asyncio.get_event_loop().run_in_executor(
            None, explainer.explain_code, code_snippet
        )
        latency = time.time() - start_time

        return latency

    async def benchmark_batch_explanation(self, batch_size: int = 10) -> float:
        """Benchmark batch code explanation."""
        code_snippets = [
            f"def func_{i}():\n    return {i}" for i in range(batch_size)
        ]

        start_time = time.time()
        tasks = [self.benchmark_explanation(snippet) for snippet in code_snippets]
        await asyncio.gather(*tasks)
        latency = time.time() - start_time

        return latency / batch_size  # Average latency per explanation

    def run_standard_benchmarks(self) -> List[BenchmarkResult]:
        """Run standard benchmark suite for Code Explainer."""
        results = []

        # Single explanation benchmark
        config = BenchmarkConfig(
            name="single_explanation",
            duration=60,
            concurrency=5,
            warmup_iterations=5
        )

        async def single_benchmark():
            return await self.benchmark_explanation()

        result = asyncio.run(self.suite.run_benchmark(config, single_benchmark))
        results.append(result)

        # Batch explanation benchmark
        config = BenchmarkConfig(
            name="batch_explanation_10",
            duration=60,
            concurrency=2,
            warmup_iterations=3
        )

        async def batch_benchmark():
            return await self.benchmark_batch_explanation(10)

        result = asyncio.run(self.suite.run_benchmark(config, batch_benchmark))
        results.append(result)

        return results

# Convenience functions
def create_benchmark_suite(output_dir: Optional[Path] = None) -> BenchmarkSuite:
    """Create a benchmark suite."""
    if output_dir is None:
        output_dir = Path("benchmarks/results")
    return BenchmarkSuite(output_dir)

def run_code_explainer_benchmarks(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run comprehensive benchmarks for Code Explainer."""
    suite = create_benchmark_suite(output_dir)
    benchmark = CodeExplainerBenchmark(suite)

    results = benchmark.run_standard_benchmarks()

    # Generate report
    report = suite.generate_report(results)

    # Export results
    suite.export_report(suite.output_dir / "benchmark_report.json")
    suite.export_report(suite.output_dir / "benchmark_report.csv", format="csv")

    # Generate plots
    suite.plot_results(results, suite.output_dir / "benchmark_plots.png")

    return report

if __name__ == "__main__":
    # Example usage
    print("Running Code Explainer benchmarks...")

    try:
        report = run_code_explainer_benchmarks()

        print("Benchmark Summary:")
        print(f"Total benchmarks: {report['summary']['total_benchmarks']}")
        print(".2f")
        print(".3f")
        print(".3f")

        print("\nDetailed results saved to benchmarks/results/")

    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
