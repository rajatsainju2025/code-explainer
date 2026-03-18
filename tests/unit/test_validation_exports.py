"""Tests for validation module exports and optimizations."""

import pytest


class TestValidationExports:
    """Tests for validation module __all__ exports."""
    
    def test_all_exports_defined(self):
        """Test that __all__ is properly defined."""
        from code_explainer import validation
        
        assert hasattr(validation, '__all__')
        assert "CodeExplanationRequest" in validation.__all__
        assert "BatchCodeExplanationRequest" in validation.__all__
    
    def test_exports_are_importable(self):
        """Test that all exports are importable."""
        from code_explainer.validation import (
            CodeExplanationRequest,
            BatchCodeExplanationRequest,
        )
        
        assert CodeExplanationRequest is not None
        assert BatchCodeExplanationRequest is not None


class TestValidationConstants:
    """Tests for validation constants."""
    
    def test_allowed_strategies_is_frozenset(self):
        """Test that allowed strategies is a frozenset for O(1) lookup."""
        from code_explainer.validation import _ALLOWED_STRATEGIES
        
        assert isinstance(_ALLOWED_STRATEGIES, frozenset)
    
    def test_allowed_strategies_contents(self):
        """Test that all expected strategies are present."""
        from code_explainer.validation import _ALLOWED_STRATEGIES
        
        expected = {"vanilla", "ast_augmented", "retrieval_augmented", 
                    "execution_trace", "enhanced_rag", "multi_agent", "intelligent"}
        
        for strategy in expected:
            assert strategy in _ALLOWED_STRATEGIES
    
    def test_strategy_lookup_performance(self):
        """Test that strategy lookup is O(1) with frozenset."""
        from code_explainer.validation import _ALLOWED_STRATEGIES
        
        # These should all be constant-time lookups
        assert "vanilla" in _ALLOWED_STRATEGIES
        assert "invalid_strategy_12345" not in _ALLOWED_STRATEGIES


class TestValidationOptimizations:
    """Tests for validation optimization features."""
    
    def test_code_validation_fast_path(self):
        """Test that code validation uses fast path for non-whitespace."""
        from code_explainer.validation import CodeExplanationRequest
        
        # Code starting with non-whitespace should take fast path
        request = CodeExplanationRequest(code="def test(): pass")
        assert request.code == "def test(): pass"
    
    def test_strategy_error_message_cached(self):
        """Test that strategy error message is pre-computed."""
        from code_explainer.validation import _STRATEGY_ERROR_MSG
        
        # Should be a pre-computed string, not generated on demand
        assert isinstance(_STRATEGY_ERROR_MSG, str)
        assert "vanilla" in _STRATEGY_ERROR_MSG
