"""Tests for new advanced features."""

import pytest
import tempfile
from pathlib import Path

from src.code_explainer.config_validator import ConfigValidator
from src.code_explainer.quality_analyzer import CodeQualityAnalyzer, IssueLevel
from src.code_explainer.profiler import PerformanceProfiler


class TestConfigValidator:
    """Test cases for configuration validation."""
    
    def test_valid_config(self, tmp_path):
        """Test validation of a valid configuration."""
        config = {
            "model": {"name": "test-model"},
            "prompting": {"strategy": "vanilla"}
        }
        
        validator = ConfigValidator()
        assert validator.validate_config(config) is True
        assert len(validator.errors) == 0
    
    def test_missing_required_section(self):
        """Test validation with missing required section."""
        config = {"model": {"name": "test-model"}}  # Missing prompting
        
        validator = ConfigValidator()
        assert validator.validate_config(config) is False
        assert len(validator.errors) == 1
        assert "Required section 'prompting' is missing" in validator.errors[0]
    
    def test_invalid_strategy(self):
        """Test validation with invalid strategy."""
        config = {
            "model": {"name": "test-model"},
            "prompting": {"strategy": "invalid_strategy"}
        }
        
        validator = ConfigValidator()
        assert validator.validate_config(config) is False
        assert any("Invalid prompting strategy" in error for error in validator.errors)
    
    def test_unknown_section_warning(self):
        """Test warning for unknown configuration section."""
        config = {
            "model": {"name": "test-model"},
            "prompting": {"strategy": "vanilla"},
            "unknown_section": {"some": "value"}
        }
        
        validator = ConfigValidator()
        assert validator.validate_config(config) is True  # Should still be valid
        assert len(validator.warnings) == 1
        assert "Unknown configuration section: 'unknown_section'" in validator.warnings[0]


class TestCodeQualityAnalyzer:
    """Test cases for code quality analysis."""
    
    def test_clean_code(self):
        """Test analysis of clean code."""
        code = '''
def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    return sum(numbers)
'''
        
        analyzer = CodeQualityAnalyzer()
        issues = analyzer.analyze_code(code)
        
        # Should have minimal issues (maybe just style preferences)
        critical_errors = [i for i in issues if i.level == IssueLevel.CRITICAL]
        assert len(critical_errors) == 0
    
    def test_dangerous_eval(self):
        """Test detection of dangerous eval usage."""
        code = '''
def bad_function(user_input):
    return eval(user_input)
'''
        
        analyzer = CodeQualityAnalyzer()
        issues = analyzer.analyze_code(code)
        
        # Should detect dangerous eval usage
        eval_issues = [i for i in issues if i.rule_id == "dangerous_eval"]
        assert len(eval_issues) == 1
        assert eval_issues[0].level == IssueLevel.CRITICAL
    
    def test_syntax_error(self):
        """Test handling of syntax errors."""
        code = '''
def broken_function(
    # Missing closing parenthesis
    pass
'''
        
        analyzer = CodeQualityAnalyzer()
        issues = analyzer.analyze_code(code)
        
        # Should detect syntax error
        syntax_errors = [i for i in issues if i.rule_id == "syntax_error"]
        assert len(syntax_errors) == 1
        assert syntax_errors[0].level == IssueLevel.ERROR
    
    def test_naming_conventions(self):
        """Test naming convention checks."""
        code = '''
def BadFunctionName():
    pass

class badClassName:
    pass

CamelCaseVariable = 42
'''
        
        analyzer = CodeQualityAnalyzer()
        issues = analyzer.analyze_code(code)
        
        # Should detect naming issues
        naming_issues = [i for i in issues if i.rule_id and "naming" in i.rule_id]
        assert len(naming_issues) >= 2  # At least function and class naming issues
    
    def test_complexity_analysis(self):
        """Test complexity analysis."""
        code = '''
def complex_function(a, b, c, d, e, f, g):  # Too many arguments
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            if g:
                                return "complex"
    return "simple"
'''
        
        analyzer = CodeQualityAnalyzer()
        issues = analyzer.analyze_code(code)
        
        # Should detect high complexity and too many arguments
        complexity_issues = [i for i in issues if i.rule_id and ("complexity" in i.rule_id or "arguments" in i.rule_id)]
        assert len(complexity_issues) >= 1
    
    def test_best_practices(self):
        """Test best practices checks."""
        code = '''
def mutable_default(items=[]):  # Mutable default argument
    items.append(1)
    return items

def old_formatting():
    return "Hello %s" % "world"  # Old string formatting
'''
        
        analyzer = CodeQualityAnalyzer()
        issues = analyzer.analyze_code(code)
        
        # Should detect mutable default and old formatting
        practice_issues = [i for i in issues if i.rule_id in ["mutable_default", "old_string_formatting"]]
        assert len(practice_issues) >= 1


class TestPerformanceProfiler:
    """Test cases for performance profiling."""
    
    def test_basic_profiling(self):
        """Test basic profiling functionality."""
        profiler = PerformanceProfiler()
        
        def test_operation():
            # Simulate some work
            sum(range(1000))
            return "done"
        
        with profiler.profile("test_operation"):
            result = test_operation()
        
        assert result == "done"
        assert len(profiler.metrics) == 1
        
        metric = profiler.metrics[0]
        assert metric.operation == "test_operation"
        assert metric.duration_ms > 0
        assert metric.memory_peak_mb > 0
    
    def test_profiling_with_metadata(self):
        """Test profiling with metadata."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("test_with_metadata", test_param="value", iterations=5):
            pass
        
        assert len(profiler.metrics) == 1
        metric = profiler.metrics[0]
        assert metric.metadata["test_param"] == "value"
        assert metric.metadata["iterations"] == 5
    
    def test_summary_generation(self):
        """Test summary generation."""
        profiler = PerformanceProfiler()
        
        # Record multiple operations
        for i in range(3):
            with profiler.profile("test_operation"):
                sum(range(100))
        
        summary = profiler.get_summary()
        
        assert "test_operation" in summary
        assert summary["test_operation"]["count"] == 3
        assert "duration_ms" in summary["test_operation"]
        assert "memory_mb" in summary["test_operation"]
    
    def test_benchmark_operation(self):
        """Test operation benchmarking."""
        profiler = PerformanceProfiler()
        
        def simple_operation(n=100):
            return sum(range(n))
        
        results = profiler.benchmark_operation(
            simple_operation,
            "simple_sum",
            num_iterations=3,
            n=50
        )
        
        assert results["operation"] == "simple_sum"
        assert results["iterations"] == 3
        assert results["successful_iterations"] == 3
        assert "duration_ms" in results
        assert len(results["results"]) == 3
    
    def test_save_and_load_metrics(self, tmp_path):
        """Test saving metrics to file."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("test_save"):
            pass
        
        metrics_file = tmp_path / "metrics.json"
        profiler.save_metrics(str(metrics_file))
        
        assert metrics_file.exists()
        
        # Verify the file contains expected data
        import json
        with open(metrics_file) as f:
            data = json.load(f)
        
        assert "summary" in data
        assert "detailed_metrics" in data
        assert len(data["detailed_metrics"]) == 1


class TestIntegration:
    """Integration tests for new features."""
    
    def test_cli_config_validation(self, tmp_path):
        """Test CLI config validation command."""
        # Create a test config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
model:
  name: "test-model"
prompting:
  strategy: "vanilla"
""")
        
        validator = ConfigValidator()
        is_valid = validator.validate_config(str(config_file))
        
        assert is_valid is True
        assert len(validator.errors) == 0
    
    def test_quality_analysis_workflow(self):
        """Test complete quality analysis workflow."""
        test_code = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
        
        analyzer = CodeQualityAnalyzer()
        issues = analyzer.analyze_code(test_code)
        summary = analyzer.get_summary()
        
        assert "total_issues" in summary
        assert "by_level" in summary
        assert "by_rule" in summary
        assert "issues" in summary
        
        # This is relatively clean code, should have minimal issues
        critical_issues = summary["by_level"]["critical"]
        assert critical_issues == 0
    
    def test_performance_profiling_workflow(self):
        """Test complete performance profiling workflow."""
        profiler = PerformanceProfiler()
        
        # Simulate a code explanation workflow
        def load_model():
            # Simulate model loading time
            import time
            time.sleep(0.01)
            return "model_loaded"
        
        def explain_code(code="def test(): pass"):
            # Simulate explanation time
            import time
            time.sleep(0.005)
            return f"Explanation for: {code}"
        
        # Profile the workflow
        with profiler.profile("model_loading"):
            model = load_model()
        
        with profiler.profile("code_explanation", code_length=len("def test(): pass")):
            explanation = explain_code()
        
        # Verify results
        assert model == "model_loaded"
        assert "Explanation for:" in explanation
        assert len(profiler.metrics) == 2
        
        summary = profiler.get_summary()
        assert "model_loading" in summary
        assert "code_explanation" in summary


if __name__ == "__main__":
    pytest.main([__file__])
