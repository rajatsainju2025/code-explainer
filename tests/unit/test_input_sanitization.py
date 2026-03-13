"""Tests for security input sanitization and validation."""

import pytest

from code_explainer.input_sanitization import InputSanitizer, InputValidator


class TestInputSanitizer:
    """Tests for InputSanitizer class."""
    
    def test_sanitize_code_valid(self):
        code = "x = 1"
        sanitized = InputSanitizer.sanitize_code(code)
        assert sanitized == "x = 1"
    
    def test_sanitize_code_empty(self):
        with pytest.raises(ValueError):
            InputSanitizer.sanitize_code("")
    
    def test_sanitize_code_whitespace_only(self):
        with pytest.raises(ValueError):
            InputSanitizer.sanitize_code("   \n\t  ")
    
    def test_sanitize_code_too_large(self):
        large_code = "x = 1\n" * 20000  # > 100KB
        with pytest.raises(ValueError, match="exceeds maximum size"):
            InputSanitizer.sanitize_code(large_code)
    
    def test_sanitize_language_valid(self):
        lang = InputSanitizer.sanitize_language("Python")
        assert lang == "python"
    
    def test_sanitize_language_invalid(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            InputSanitizer.sanitize_language("Brainfuck")
    
    def test_sanitize_language_case_insensitive(self):
        lang = InputSanitizer.sanitize_language("JAVASCRIPT")
        assert lang == "javascript"
    
    def test_sanitize_string_valid(self):
        s = InputSanitizer.sanitize_string("test")
        assert s == "test"
    
    def test_sanitize_string_too_long(self):
        long_string = "a" * 2000
        with pytest.raises(ValueError, match="exceeds maximum length"):
            InputSanitizer.sanitize_string(long_string)
    
    def test_sanitize_string_not_string(self):
        with pytest.raises(ValueError):
            InputSanitizer.sanitize_string(123)
    
    def test_sanitize_string_whitespace_disallowed(self):
        with pytest.raises(ValueError, match="Whitespace not allowed"):
            InputSanitizer.sanitize_string("has space", allow_whitespace=False)
    
    def test_sanitize_integer_valid(self):
        num = InputSanitizer.sanitize_integer(42)
        assert num == 42
    
    def test_sanitize_integer_from_string(self):
        num = InputSanitizer.sanitize_integer("42")
        assert num == 42
    
    def test_sanitize_integer_below_min(self):
        with pytest.raises(ValueError, match="below minimum"):
            InputSanitizer.sanitize_integer(-5, min_val=0)
    
    def test_sanitize_integer_above_max(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            InputSanitizer.sanitize_integer(100, max_val=50)
    
    def test_sanitize_integer_invalid(self):
        with pytest.raises(ValueError, match="not a valid integer"):
            InputSanitizer.sanitize_integer("not a number")
    
    def test_sanitize_float_valid(self):
        num = InputSanitizer.sanitize_float(3.14)
        assert num == 3.14
    
    def test_sanitize_float_from_string(self):
        num = InputSanitizer.sanitize_float("3.14")
        assert num == 3.14
    
    def test_sanitize_float_range(self):
        num = InputSanitizer.sanitize_float(0.5, min_val=0.0, max_val=1.0)
        assert num == 0.5


class TestInputValidator:
    """Tests for InputValidator class."""
    
    def test_validate_code_request_valid(self):
        code, lang = InputValidator.validate_code_request("x = 1", "python")
        assert code == "x = 1"
        assert lang == "python"
    
    def test_validate_code_request_empty_code(self):
        with pytest.raises(ValueError):
            InputValidator.validate_code_request("", "python")
    
    def test_validate_code_request_invalid_language(self):
        with pytest.raises(ValueError):
            InputValidator.validate_code_request("x = 1", "invalid")
    
    def test_validate_batch_request_valid(self):
        codes = InputValidator.validate_batch_request(["x = 1", "y = 2"])
        assert len(codes) == 2
    
    def test_validate_batch_request_not_list(self):
        with pytest.raises(ValueError):
            InputValidator.validate_batch_request("not a list")
    
    def test_validate_batch_request_empty(self):
        with pytest.raises(ValueError):
            InputValidator.validate_batch_request([])
    
    def test_validate_batch_request_too_large(self):
        codes = ["x = 1"] * 101
        with pytest.raises(ValueError, match="cannot exceed 100"):
            InputValidator.validate_batch_request(codes)
