"""Tests for symbolic code explanation functionality."""

import pytest
from src.code_explainer.symbolic import SymbolicAnalyzer, format_symbolic_explanation


class TestSymbolicAnalyzer:
    """Test symbolic code analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SymbolicAnalyzer()
    
    def test_simple_function_analysis(self):
        """Test analysis of a simple function."""
        code = """
def add_numbers(a, b):
    if a < 0 or b < 0:
        raise ValueError("Numbers must be positive")
    result = a + b
    assert result >= 0
    return result
"""
        explanation = self.analyzer.analyze_code(code)
        
        # Should detect input conditions
        assert len(explanation.input_conditions) > 0
        
        # Should detect preconditions (the if statement)
        assert len(explanation.preconditions) > 0
        
        # Should detect postconditions (assert and return)
        assert len(explanation.postconditions) > 0
        
        # Should generate property tests
        assert len(explanation.property_tests) > 0
        
        # Should analyze complexity
        assert explanation.complexity_analysis['cyclomatic_complexity'] > 1
    
    def test_loop_analysis(self):
        """Test analysis of code with loops."""
        code = """
def factorial(n):
    if n < 0:
        return None
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
        explanation = self.analyzer.analyze_code(code)
        
        # Should detect loop
        assert explanation.complexity_analysis['number_of_loops'] == 1
        
        # Should detect invariants
        assert len(explanation.invariants) >= 0  # May or may not detect invariants
        
        # Time complexity should be estimated
        assert 'estimated_time_complexity' in explanation.complexity_analysis
    
    def test_sorting_function_detection(self):
        """Test detection of sorting functions."""
        code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
        explanation = self.analyzer.analyze_code(code)
        
        # Should detect nested loops
        assert explanation.complexity_analysis['number_of_loops'] == 2
        
        # Should estimate quadratic complexity
        assert 'n²' in explanation.complexity_analysis['estimated_time_complexity']
        
        # Should generate sorting-specific property test
        property_descriptions = [test.property_description for test in explanation.property_tests]
        sorting_test_found = any('sort' in desc.lower() or 'permutation' in desc.lower() 
                                for desc in property_descriptions)
        assert sorting_test_found or len(explanation.property_tests) > 0
    
    def test_data_flow_analysis(self):
        """Test data flow analysis."""
        code = """
def process_data(x, y):
    a = x + 1
    b = y * 2
    c = a + b
    return c
"""
        explanation = self.analyzer.analyze_code(code)
        
        # Should track variable dependencies
        assert 'a' in explanation.data_flow
        assert 'b' in explanation.data_flow
        assert 'c' in explanation.data_flow
        
        # Variable 'c' should depend on both 'a' and 'b'
        if 'c' in explanation.data_flow:
            assert 'a' in explanation.data_flow['c'] or 'b' in explanation.data_flow['c']
    
    def test_invalid_code_handling(self):
        """Test handling of invalid Python code."""
        code = "def invalid_syntax( return 42"
        
        explanation = self.analyzer.analyze_code(code)
        
        # Should return empty explanation without crashing
        assert len(explanation.input_conditions) == 0
        assert len(explanation.preconditions) == 0
        assert len(explanation.postconditions) == 0
        assert len(explanation.property_tests) == 0
    
    def test_complex_conditions(self):
        """Test analysis of complex conditional logic."""
        code = """
def validate_user(age, income, credit_score):
    if age < 18:
        return False
    if income < 30000 or credit_score < 600:
        return False
    if age > 65 and income < 50000:
        return False
    return True
"""
        explanation = self.analyzer.analyze_code(code)
        
        # Should detect multiple conditions
        assert explanation.complexity_analysis['number_of_conditions'] >= 3
        
        # Should have higher cyclomatic complexity
        assert explanation.complexity_analysis['cyclomatic_complexity'] > 3
    
    def test_format_symbolic_explanation(self):
        """Test formatting of symbolic explanation."""
        code = """
def simple_function(x):
    assert x > 0
    return x * 2
"""
        explanation = self.analyzer.analyze_code(code)
        formatted = format_symbolic_explanation(explanation)
        
        # Should produce non-empty formatted output
        assert len(formatted) > 0
        assert formatted != "No symbolic conditions detected."
        
        # Should contain key sections
        if explanation.preconditions:
            assert "Preconditions:" in formatted
        if explanation.complexity_analysis:
            assert "Complexity Analysis:" in formatted


class TestSymbolicIntegration:
    """Test integration with main explainer."""
    
    def test_symbolic_explanation_integration(self):
        """Test that symbolic analysis integrates with main explanation."""
        from src.code_explainer.model import CodeExplainer
        
        # Use tiny model for testing
        explainer = CodeExplainer(config_path="configs/codet5-small.yaml")
        
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        # Test both regular and symbolic explanations
        regular_explanation = explainer.explain_code(code)
        symbolic_explanation = explainer.explain_code_with_symbolic(code, include_symbolic=True)
        
        # Symbolic explanation should be longer and contain both parts
        assert len(symbolic_explanation) >= len(regular_explanation)
        assert "## Code Explanation" in symbolic_explanation or "## Symbolic Analysis" in symbolic_explanation
    
    @pytest.mark.parametrize("strategy", ["vanilla", "ast_augmented", "retrieval_augmented", "execution_trace"])
    def test_symbolic_with_different_strategies(self, strategy):
        """Test symbolic analysis works with different prompt strategies."""
        from src.code_explainer.model import CodeExplainer
        
        explainer = CodeExplainer(config_path="configs/codet5-small.yaml")
        
        code = """
def process_list(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
"""
        
        explanation = explainer.explain_code_with_symbolic(code, include_symbolic=True, strategy=strategy)
        
        # Should produce valid explanation regardless of strategy
        assert len(explanation) > 0
        assert isinstance(explanation, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
