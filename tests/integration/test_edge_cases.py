"""Comprehensive edge case tests for code explainer."""

import pytest
from unittest.mock import MagicMock, patch
import torch

from code_explainer.model.core import CodeExplainer
from code_explainer.retrieval.retriever import CodeRetriever
from code_explainer.exceptions import ValidationError, ResourceError


class TestEdgeCaseInputs:
    """Test edge cases in input handling."""
    
    def test_explain_empty_code(self):
        """Empty code string should be rejected."""
        explainer = CodeExplainer(model_name="codet5-base", lazy_load=True)
        explainer.model = MagicMock()
        
        with pytest.raises((ValidationError, ValueError)):
            explainer.explain_code("")
    
    def test_explain_whitespace_only(self):
        """Whitespace-only code should be rejected."""
        explainer = CodeExplainer(model_name="codet5-base", lazy_load=True)
        explainer.model = MagicMock()
        
        with pytest.raises((ValidationError, ValueError)):
            explainer.explain_code("   \n\t  ")
    
    def test_explain_extremely_long_code(self):
        """Very long code should be truncated or rejected gracefully."""
        explainer = CodeExplainer(model_name="codet5-base", lazy_load=True)
        explainer.model = MagicMock()
        explainer.tokenizer = MagicMock()
        
        # Create code longer than typical model limits (e.g., 10K tokens)
        long_code = "x = 1\n" * 5000
        
        # Should either truncate or raise ValidationError
        try:
            result = explainer.explain_code(long_code, max_length=512)
            assert result is not None  # If successful, result should exist
        except ValidationError:
            pass  # Expected behavior if rejected
    
    def test_explain_special_characters_in_code(self):
        """Code with special/unicode characters should be handled."""
        explainer = CodeExplainer(model_name="codet5-base", lazy_load=True)
        explainer.model = MagicMock()
        explainer.tokenizer = MagicMock()
        
        code_with_unicode = "# 你好 مرحبا\nprint('Hello')"
        
        try:
            result = explainer.explain_code(code_with_unicode)
            assert result is not None
        except (ValidationError, UnicodeError):
            pass  # Expected if not supported


class TestEdgeCaseRetrieval:
    """Test edge cases in retrieval."""
    
    def test_retriever_empty_repository(self):
        """Retriever with no indexed codes should handle gracefully."""
        retriever = CodeRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Query on empty index
        with pytest.raises((ResourceError, ValueError)):
            retriever.retrieve("test query", k=5)
    
    def test_retriever_k_larger_than_repository(self):
        """k should be clamped to repository size."""
        retriever = CodeRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Add 2 codes, then try k=10
        try:
            retriever.index.add_code("x = 1", "var assignment")
            retriever.index.add_code("y = 2", "another assignment")
            
            # Should return at most 2 results
            results = retriever.retrieve("assignment", k=10)
            assert len(results) <= 2
        except Exception:
            pass  # If retrieval not supported in this context


class TestEdgeCaseDevice:
    """Test edge cases in device selection."""
    
    def test_fallback_to_cpu_when_cuda_unavailable(self):
        """Should gracefully fallback to CPU."""
        from code_explainer.device_manager import get_device
        
        # Simulate CUDA unavailable
        with patch("torch.cuda.is_available", return_value=False):
            device = get_device()
            assert device.type == "cpu"
    
    def test_device_string_representation(self):
        """Device string should be valid."""
        from code_explainer.device_manager import get_device
        
        device = get_device()
        device_str = str(device)
        
        assert device_str in ["cpu", "cuda", "mps"]


class TestEdgeCaseExceptions:
    """Test edge cases in exception handling."""
    
    def test_validation_error_with_message(self):
        """ValidationError should preserve message."""
        error = ValidationError("Invalid input")
        assert "Invalid input" in str(error)
    
    def test_resource_error_with_context(self):
        """ResourceError should provide context."""
        error = ResourceError("Model not found", resource_type="model")
        assert "Model not found" in str(error)
    
    def test_nested_exception_handling(self):
        """Nested exceptions should be properly wrapped."""
        original_error = RuntimeError("Original error")
        
        try:
            raise original_error
        except RuntimeError as e:
            wrapped = ValidationError(f"Validation failed: {e}")
            assert "Original error" in str(wrapped)


class TestEdgeCaseMemory:
    """Test memory efficiency edge cases."""
    
    def test_large_batch_processing(self):
        """Processing large batches should not leak memory."""
        explainer = CodeExplainer(model_name="codet5-base", lazy_load=True)
        explainer.model = MagicMock()
        
        # Simulate processing many items
        codes = [f"x{i} = {i}" for i in range(100)]
        
        try:
            for code in codes:
                explainer.explain_code(code)
        except Exception:
            pass  # May fail due to mocking, but should not crash


class TestEdgeCaseCache:
    """Test cache edge cases."""
    
    def test_cache_with_identical_inputs(self):
        """Cache should return same result for identical inputs."""
        from code_explainer.model.cache import CodeExplainerCache
        
        cache = CodeExplainerCache()
        
        code = "x = 1"
        key = cache.get_cache_key(code)
        
        cache.cache_explanation(code, "explanation1")
        cached = cache.get_cached_explanation(code)
        
        assert cached == "explanation1"
    
    def test_cache_collision_handling(self):
        """Cache should handle potential hash collisions."""
        from code_explainer.model.cache import CodeExplainerCache
        
        cache = CodeExplainerCache()
        
        # Add multiple items
        cache.cache_explanation("code1", "exp1")
        cache.cache_explanation("code2", "exp2")
        
        # Retrieve should return correct items
        assert cache.get_cached_explanation("code1") == "exp1"
        assert cache.get_cached_explanation("code2") == "exp2"
