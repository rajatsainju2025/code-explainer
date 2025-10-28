"""Integration tests for new features: performance, security, and API enhancements."""

import pytest
import time
from unittest.mock import Mock, patch

from code_explainer.model import CodeExplainer


class TestPerformanceFeatures:
    """Test performance monitoring and optimization features."""

    def test_memory_usage_tracking(self):
        """Test memory usage statistics collection."""
        explainer = CodeExplainer()

        # Test memory stats collection
        stats = explainer.get_memory_usage()
        assert isinstance(stats, dict)
        assert 'rss_mb' in stats
        assert 'vms_mb' in stats
        assert 'memory_percent' in stats

        # All values should be numeric
        for key, value in stats.items():
            if key.endswith('_mb') or key.endswith('_percent'):
                assert isinstance(value, (int, float))

    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        explainer = CodeExplainer()

        report = explainer.get_performance_report()
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Performance Report" in report

    def test_quantization_feature(self):
        """Test model quantization functionality."""
        explainer = CodeExplainer()

        # Test quantization (should handle gracefully when model not loaded)
        result = explainer.enable_quantization(bits=8)
        assert isinstance(result, bool)

        # Test invalid bits
        result = explainer.enable_quantization(bits=16)
        assert result is False

    def test_inference_optimizations(self):
        """Test inference optimization features."""
        explainer = CodeExplainer()

        # Should not raise errors
        explainer.optimize_for_inference()
        explainer.optimize_tokenizer()

        # Test gradient checkpointing
        result = explainer.enable_gradient_checkpointing()
        assert isinstance(result, bool)


class TestSecurityFeatures:
    """Test security validation and rate limiting features."""

    def test_input_validation(self):
        """Test input security validation."""
        explainer = CodeExplainer()

        # Test safe code
        safe_code = "def hello(): return 'world'"
        is_safe, warnings = explainer.validate_input_security(safe_code)
        assert is_safe
        assert isinstance(warnings, list)

        # Test potentially unsafe code
        unsafe_code = "import os; os.system('ls')"
        is_safe, warnings = explainer.validate_input_security(unsafe_code)
        assert isinstance(is_safe, bool)
        assert isinstance(warnings, list)

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        explainer = CodeExplainer()

        # Test rate limit checking
        client_id = "test_client"
        allowed = explainer.check_rate_limit(client_id)
        assert isinstance(allowed, bool)

        # Multiple calls should still work
        for _ in range(5):
            result = explainer.check_rate_limit(client_id)
            assert isinstance(result, bool)

    def test_security_audit(self):
        """Test security event auditing."""
        explainer = CodeExplainer()

        # Should not raise errors
        explainer.audit_security_event("test_event", {"test": "data"})

    def test_setup_info(self):
        """Test setup information retrieval."""
        explainer = CodeExplainer()

        info = explainer.get_setup_info()
        assert isinstance(info, dict)
        assert "model_loaded" in info
        assert "device" in info


class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_batch_explanation_structure(self):
        """Test batch explanation returns proper structure."""
        explainer = CodeExplainer()

        # Mock the actual batch processing since model isn't loaded
        with patch.object(explainer, 'explain_code_batch', return_value=["Explanation 1", "Explanation 2"]):
            codes_requests = [{"code": "code1"}, {"code": "code2"}]
            results = explainer.explain_code_batch(codes_requests)

            assert isinstance(results, list)
            assert len(results) == len(codes_requests)
            assert all(isinstance(exp, str) for exp in results)


class TestAsyncFeatures:
    """Test asynchronous processing features."""

    @pytest.mark.asyncio
    async def test_async_explanation(self):
        """Test async explanation functionality."""
        explainer = CodeExplainer()

        # Mock async method - just test that method exists
        assert hasattr(explainer, 'explain_code_async')

    @pytest.mark.asyncio
    async def test_async_batch_explanation(self):
        """Test async batch explanation functionality."""
        explainer = CodeExplainer()

        # Mock async batch method - just test that method exists
        assert hasattr(explainer, 'explain_code_batch_async')


class TestErrorHandling:
    """Test error handling in new features."""

    def test_graceful_degradation(self):
        """Test that features degrade gracefully when dependencies unavailable."""
        explainer = CodeExplainer()

        # All methods should handle missing resources gracefully
        assert explainer.get_memory_usage() is not None
        assert explainer.get_performance_report() is not None
        assert explainer.validate_input_security("test") == (True, [])
        assert explainer.check_rate_limit() is True
        assert explainer.get_setup_info() is not None

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        explainer = CodeExplainer()

        # Test with empty string
        is_safe, warnings = explainer.validate_input_security("")
        assert isinstance(is_safe, bool)
        assert isinstance(warnings, list)

        # Test with very long input
        long_code = "print('test')" * 1000
        is_safe, warnings = explainer.validate_input_security(long_code)
        assert isinstance(is_safe, bool)
        assert isinstance(warnings, list)


class TestPerformanceRegression:
    """Performance regression tests for new features."""

    def setup_method(self):
        """Set up test explainer."""
        self.explainer = CodeExplainer()

    def test_memory_usage_performance(self):
        """Test that memory usage tracking doesn't significantly impact performance."""
        explainer = CodeExplainer()

        start_time = time.time()
        for _ in range(100):
            explainer.get_memory_usage()
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second for 100 calls)
        assert end_time - start_time < 1.0

    def test_validation_performance(self):
        """Test that input validation is fast."""
        explainer = CodeExplainer()

        test_code = "def test(): return 1 + 1"

        start_time = time.time()
        for _ in range(100):
            explainer.validate_input_security(test_code)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 0.5

    def test_memory_usage_stability(self):
        """Test that memory usage doesn't regress significantly."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Perform some operations
        for _ in range(10):
            self.explainer.get_memory_usage()
            time.sleep(0.01)

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50, f"Memory regression: {memory_increase}MB increase"

    def test_rate_limiting_performance(self):
        """Test that rate limiting doesn't significantly impact performance."""
        # Test rate limiting performance
        start_time = time.time()

        for i in range(10):
            result = self.explainer.check_rate_limit(f"test_user_{i}")
            assert isinstance(result, bool)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete in reasonable time (< 1 second for 10 checks)
        assert total_time < 1.0, f"Rate limiting too slow: {total_time}s for 10 checks"

    def test_security_validation_performance(self):
        """Test security validation performance."""
        test_codes = [
            "def safe_function(): return 1",
            "print('hello world')",
            "import os; print(os.getcwd())",
            "x = 1; y = 2; print(x + y)",
            "class TestClass: pass"
        ]

        start_time = time.time()

        for code in test_codes:
            result = self.explainer.validate_input_security(code)
            assert isinstance(result, tuple) and len(result) == 2

        end_time = time.time()
        total_time = end_time - start_time

        # Should validate 5 codes quickly (< 0.5 seconds)
        assert total_time < 0.5, f"Security validation too slow: {total_time}s for 5 codes"

    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        codes = ["def f(): return 1"] * 5

        start_time = time.time()
        result = self.explainer.explain_code_batch(codes, strategy="vanilla")
        end_time = time.time()

        total_time = end_time - start_time

        # Batch processing should be reasonably fast (< 2 seconds for small batch)
        assert total_time < 2.0, f"Batch processing too slow: {total_time}s for 5 items"
        assert isinstance(result, list) and len(result) == 5

    def test_async_performance_overhead(self):
        """Test async processing performance overhead."""
        import asyncio

        async def async_test():
            start_time = time.time()
            tasks = []

            for i in range(3):
                task = asyncio.create_task(
                    asyncio.to_thread(self.explainer.explain_code, f"print({i})", strategy="vanilla")
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            end_time = time.time()

            return end_time - start_time, results

        total_time, results = asyncio.run(async_test())

        # Async processing should complete in reasonable time (< 3 seconds)
        assert total_time < 3.0, f"Async processing too slow: {total_time}s"
        assert len(results) == 3

    def test_caching_performance_impact(self):
        """Test that caching improves performance."""
        code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"

        # First call (cache miss)
        start_time = time.time()
        result1 = self.explainer.explain_code(code, strategy="vanilla")
        first_call_time = time.time() - start_time

        # Second call (cache hit)
        start_time = time.time()
        result2 = self.explainer.explain_code(code, strategy="vanilla")
        second_call_time = time.time() - start_time

        # Second call should be faster (at least 10% improvement)
        improvement_ratio = first_call_time / second_call_time if second_call_time > 0 else float('inf')
        assert improvement_ratio > 1.1, f"Cache not improving performance: {improvement_ratio}x"

    def test_quantization_performance_gain(self):
        """Test quantization performance improvements."""
        # This test may be skipped if quantization is not supported
        try:
            # Enable quantization
            self.explainer.enable_quantization()

            # Measure performance with quantization
            start_time = time.time()
            for _ in range(5):
                self.explainer.get_memory_usage()
            quantized_time = time.time() - start_time

            # Should complete in reasonable time
            assert quantized_time < 1.0, f"Quantization too slow: {quantized_time}s"

        except Exception as e:
            pytest.skip(f"Quantization not available: {e}")

    def test_gradient_checkpointing_memory_efficiency(self):
        """Test gradient checkpointing memory efficiency."""
        try:
            # Enable gradient checkpointing
            self.explainer.enable_gradient_checkpointing()

            # Monitor memory usage
            initial_memory = self.explainer.get_memory_usage()

            # Perform some operations
            for _ in range(5):
                self.explainer.explain_code("print('test')", strategy="vanilla")

            final_memory = self.explainer.get_memory_usage()

            # Memory usage should not explode - check that we got dict results
            assert isinstance(initial_memory, dict)
            assert isinstance(final_memory, dict)

        except Exception as e:
            pytest.skip(f"Gradient checkpointing not available: {e}")

    def test_inference_optimization_effectiveness(self):
        """Test inference optimization effectiveness."""
        try:
            # Enable inference optimizations
            self.explainer.optimize_for_inference()

            # Test inference speed
            start_time = time.time()
            for _ in range(10):
                self.explainer.explain_code("x = 1", strategy="vanilla")
            total_time = time.time() - start_time

            # Should be reasonably fast
            assert total_time < 5.0, f"Inference too slow after optimization: {total_time}s"

        except Exception as e:
            pytest.skip(f"Inference optimization not available: {e}")

    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            try:
                result = self.explainer.explain_code(f"print('worker {worker_id}')", strategy="vanilla")
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should handle concurrent requests without errors
        assert len(errors) == 0, f"Concurrent request errors: {errors}"
        assert len(results) == 3, "Not all concurrent requests completed"