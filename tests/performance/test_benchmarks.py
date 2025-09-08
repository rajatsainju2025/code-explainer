"""
Performance and load testing for the Code Explainer system.

Tests system behavior under various load conditions and
identifies performance bottlenecks.
"""

import pytest
import time
import concurrent.futures
import threading
from pathlib import Path
import tempfile

from evals.config import EvalConfig
from evals.runner import EvalRunner
from evals.metrics import MetricsCalculator


class TestPerformanceBenchmarks:
    """Performance benchmarks for critical system components."""
    
    @pytest.mark.benchmark
    def test_metrics_calculation_speed(self, benchmark):
        """Benchmark metrics calculation performance."""
        calculator = MetricsCalculator(bootstrap_samples=100)
        
        # Large dataset for benchmarking
        predictions = ['test prediction'] * 1000
        references = ['test reference'] * 1000
        latencies = [0.1] * 1000
        costs = [0.01] * 1000
        
        def calculate_metrics():
            return calculator.calculate_all_metrics(
                predictions, references, latencies, costs
            )
        
        result = benchmark(calculate_metrics)
        assert result.num_samples == 1000
    
    @pytest.mark.benchmark
    def test_config_loading_speed(self, benchmark):
        """Benchmark configuration loading performance."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
name: "benchmark_config"
seed: 42
model:
  name: "test-model"
  temperature: 0.1
retrieval:
  enabled: true
  top_k: 10
""")
            f.flush()
            
            from evals.config import load_config
            
            def load_config_func():
                return load_config(f.name)
            
            config = benchmark(load_config_func)
            assert config.name == "benchmark_config"
    
    def test_concurrent_evaluations(self):
        """Test concurrent evaluation performance."""
        def run_single_eval(eval_id):
            with tempfile.TemporaryDirectory() as tmpdir:
                config = EvalConfig(
                    name=f"concurrent_eval_{eval_id}",
                    output_dir=f"{tmpdir}/eval_{eval_id}",
                    seed=42 + eval_id
                )
                config.dataset.max_samples = 5  # Small dataset
                
                runner = EvalRunner(config)
                start_time = time.time()
                results = runner.run_evaluation()
                end_time = time.time()
                
                return {
                    'eval_id': eval_id,
                    'duration': end_time - start_time,
                    'num_samples': results.num_samples
                }
        
        # Run multiple evaluations concurrently
        num_concurrent = 3
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(run_single_eval, i) for i in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # All evaluations should complete
        assert len(results) == num_concurrent
        
        # Concurrent execution should be faster than sequential
        avg_duration = sum(r['duration'] for r in results) / len(results)
        assert total_duration < avg_duration * num_concurrent * 0.8  # 20% speedup threshold
    
    def test_memory_stress(self):
        """Test system behavior under memory stress."""
        calculator = MetricsCalculator(bootstrap_samples=10)
        
        # Gradually increase dataset size
        max_size = 5000
        step_size = 1000
        
        for size in range(step_size, max_size + 1, step_size):
            predictions = [f'prediction_{i}' for i in range(size)]
            references = [f'reference_{i}' for i in range(size)]
            latencies = [0.1] * size
            costs = [0.01] * size
            
            start_time = time.time()
            results = calculator.calculate_all_metrics(
                predictions, references, latencies, costs
            )
            end_time = time.time()
            
            assert results.num_samples == size
            
            # Performance should degrade gracefully
            duration = end_time - start_time
            assert duration < size * 0.001  # 1ms per sample threshold
    
    def test_evaluation_scaling(self):
        """Test evaluation performance scaling with dataset size."""
        sizes = [10, 50, 100, 500]
        durations = []
        
        for size in sizes:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = EvalConfig(
                    name=f"scale_test_{size}",
                    output_dir=f"{tmpdir}/scale_{size}",
                    seed=42
                )
                config.dataset.max_samples = size
                
                runner = EvalRunner(config)
                start_time = time.time()
                results = runner.run_evaluation()
                end_time = time.time()
                
                duration = end_time - start_time
                durations.append(duration)
                
                assert results.num_samples == size
        
        # Check that scaling is roughly linear (not exponential)
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            duration_ratio = durations[i] / durations[i-1]
            
            # Duration increase should be proportional to size increase
            # Allow some overhead, so check it's less than quadratic
            assert duration_ratio < size_ratio * 2


class TestLoadTesting:
    """Load testing for API and system endpoints."""
    
    def test_rapid_fire_requests(self):
        """Test handling of rapid successive requests."""
        calculator = MetricsCalculator(bootstrap_samples=0)  # Fast mode
        
        num_requests = 100
        request_delay = 0.01  # 10ms between requests
        
        def make_request(request_id):
            predictions = [f'prediction_{request_id}']
            references = [f'reference_{request_id}']
            latencies = [0.1]
            costs = [0.01]
            
            return calculator.calculate_all_metrics(
                predictions, references, latencies, costs
            )
        
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            result = make_request(i)
            results.append(result)
            time.sleep(request_delay)
        
        end_time = time.time()
        
        # All requests should complete successfully
        assert len(results) == num_requests
        
        # Check response times are reasonable
        total_duration = end_time - start_time
        avg_response_time = total_duration / num_requests
        assert avg_response_time < 0.1  # 100ms average response time
    
    def test_thread_safety(self):
        """Test thread safety of core components."""
        calculator = MetricsCalculator(bootstrap_samples=10)
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                for i in range(10):
                    predictions = [f'thread_{thread_id}_pred_{i}']
                    references = [f'thread_{thread_id}_ref_{i}']
                    latencies = [0.1]
                    costs = [0.01]
                    
                    result = calculator.calculate_all_metrics(
                        predictions, references, latencies, costs
                    )
                    results.append((thread_id, i, result))
                    
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run multiple threads concurrently
        threads = []
        num_threads = 5
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == num_threads * 10
        
        # Verify all results are valid
        for thread_id, iteration, result in results:
            assert result.num_samples == 1
            assert result.avg_latency == 0.1


class TestRegressionDetection:
    """Tests for detecting performance regressions."""
    
    def test_baseline_performance_metrics(self):
        """Establish baseline performance metrics."""
        # This test should be run regularly to detect regressions
        calculator = MetricsCalculator(bootstrap_samples=100)
        
        # Standard test dataset
        size = 100
        predictions = [f'prediction_{i}' for i in range(size)]
        references = [f'reference_{i}' for i in range(size)]
        latencies = [0.1] * size
        costs = [0.01] * size
        
        start_time = time.time()
        results = calculator.calculate_all_metrics(
            predictions, references, latencies, costs
        )
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Baseline performance thresholds
        assert duration < 2.0, f"Metrics calculation took {duration:.2f}s, exceeds 2s threshold"
        assert results.num_samples == size
        assert results.avg_latency == 0.1
        
        # Store baseline metrics for comparison
        baseline_metrics = {
            'duration': duration,
            'size': size,
            'throughput': size / duration
        }
        
        # In a real implementation, you'd save this to a file or database
        # for comparison in future runs
        print(f"Baseline metrics: {baseline_metrics}")
    
    def test_memory_baseline(self):
        """Test memory usage baseline."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run memory-intensive operation
        calculator = MetricsCalculator(bootstrap_samples=1000)
        size = 1000
        predictions = [f'prediction_{i}' for i in range(size)]
        references = [f'reference_{i}' for i in range(size)]
        latencies = [0.1] * size
        costs = [0.01] * size
        
        results = calculator.calculate_all_metrics(
            predictions, references, latencies, costs
        )
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increase: {memory_increase / 1024 / 1024:.1f}MB"
        
        print(f"Memory usage: {memory_increase / 1024 / 1024:.1f}MB increase")


class TestStressConditions:
    """Tests under stress conditions and edge cases."""
    
    def test_extremely_large_inputs(self):
        """Test handling of extremely large inputs."""
        calculator = MetricsCalculator(bootstrap_samples=0)  # Skip bootstrap for speed
        
        # Very large text inputs
        large_prediction = "This is a very long prediction. " * 1000  # ~30KB text
        large_reference = "This is a very long reference. " * 1000
        
        predictions = [large_prediction]
        references = [large_reference]
        latencies = [1.0]
        costs = [0.1]
        
        start_time = time.time()
        results = calculator.calculate_all_metrics(
            predictions, references, latencies, costs
        )
        end_time = time.time()
        
        # Should complete in reasonable time even with large inputs
        assert end_time - start_time < 10.0
        assert results.num_samples == 1
    
    def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion scenarios."""
        calculator = MetricsCalculator(bootstrap_samples=10)
        
        # Simulate resource exhaustion by creating many large objects
        large_objects = []
        try:
            # Create increasingly large datasets until we hit memory pressure
            for size in [1000, 2000, 5000, 10000]:
                predictions = [f'pred_{i}' * 100 for i in range(size)]
                references = [f'ref_{i}' * 100 for i in range(size)]
                latencies = [0.1] * size
                costs = [0.01] * size
                
                # Store objects to simulate memory pressure
                large_objects.append((predictions, references, latencies, costs))
                
                # Try to calculate metrics
                results = calculator.calculate_all_metrics(
                    predictions, references, latencies, costs
                )
                assert results.num_samples == size
                
        except MemoryError:
            # If we hit memory limits, that's expected
            pass
        finally:
            # Clean up
            large_objects.clear()
        
        # System should recover and work normally after cleanup
        simple_predictions = ['test']
        simple_references = ['reference']
        simple_latencies = [0.1]
        simple_costs = [0.01]
        
        recovery_results = calculator.calculate_all_metrics(
            simple_predictions, simple_references, simple_latencies, simple_costs
        )
        assert recovery_results.num_samples == 1


if __name__ == '__main__':
    # Run performance tests
    pytest.main([
        __file__,
        '-v',
        '--benchmark-only',
        '--benchmark-sort=mean'
    ])
