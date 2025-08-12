"""Tests for security functionality."""

import pytest

from code_explainer.security import CodeSecurityValidator, SafeCodeExecutor, hash_code, sanitize_code_for_display


class TestCodeSecurityValidator:
    """Test cases for code security validation."""

    def test_safe_code(self):
        """Test validation of safe code."""
        validator = CodeSecurityValidator()
        
        safe_code = """
def add(a, b):
    return a + b

result = add(1, 2)
print(result)
"""
        
        is_safe, issues = validator.validate_code(safe_code)
        assert is_safe
        assert len(issues) == 0

    def test_dangerous_imports(self):
        """Test detection of dangerous imports."""
        validator = CodeSecurityValidator()
        
        dangerous_code = """
import os
os.system('rm -rf /')
"""
        
        is_safe, issues = validator.validate_code(dangerous_code)
        assert not is_safe
        assert any("import" in issue.lower() for issue in issues)

    def test_dangerous_functions(self):
        """Test detection of dangerous function calls."""
        validator = CodeSecurityValidator()
        
        dangerous_code = """
eval("__import__('os').system('rm -rf /')")
"""
        
        is_safe, issues = validator.validate_code(dangerous_code)
        assert not is_safe
        assert any("eval" in issue.lower() for issue in issues)

    def test_file_operations(self):
        """Test detection of file operations."""
        validator = CodeSecurityValidator()
        
        file_code = """
with open('/etc/passwd', 'r') as f:
    content = f.read()
"""
        
        is_safe, issues = validator.validate_code(file_code)
        assert not is_safe
        assert any("file" in issue.lower() for issue in issues)

    def test_network_operations(self):
        """Test detection of network operations."""
        validator = CodeSecurityValidator()
        
        network_code = """
import requests
response = requests.get('http://evil.com')
"""
        
        is_safe, issues = validator.validate_code(network_code)
        assert not is_safe
        assert any("network" in issue.lower() or "import" in issue.lower() for issue in issues)

    def test_strict_mode(self):
        """Test strict mode validation."""
        validator = CodeSecurityValidator(strict_mode=True)
        
        code = """
import math
result = math.sqrt(16)
"""
        
        # Should be more restrictive in strict mode
        is_safe, issues = validator.validate_code(code)
        # math module should be safe even in strict mode
        assert is_safe

    def test_syntax_error(self):
        """Test handling of syntax errors."""
        validator = CodeSecurityValidator()
        
        invalid_code = "def incomplete_function("
        
        is_safe, issues = validator.validate_code(invalid_code)
        assert not is_safe
        assert any("syntax" in issue.lower() for issue in issues)


class TestSafeCodeExecutor:
    """Test cases for safe code execution."""

    def test_safe_execution(self):
        """Test safe execution of simple code."""
        executor = SafeCodeExecutor()
        
        safe_code = """
result = 1 + 2
print(f"Result: {result}")
"""
        
        result = executor.execute_code(safe_code)
        assert result["success"]
        assert "Result: 3" in result["output"]

    def test_security_rejection(self):
        """Test rejection of dangerous code."""
        executor = SafeCodeExecutor()
        
        dangerous_code = """
import os
os.system('echo "hacked"')
"""
        
        result = executor.execute_code(dangerous_code)
        assert not result["success"]
        assert "security validation failed" in result["error"].lower()

    def test_timeout_handling(self):
        """Test timeout handling for long-running code."""
        executor = SafeCodeExecutor(timeout=1)
        
        long_running_code = """
import time
time.sleep(5)
print("Should not reach here")
"""
        
        result = executor.execute_code(long_running_code)
        assert not result["success"]
        assert "timeout" in result["error"].lower()

    def test_syntax_error_handling(self):
        """Test handling of syntax errors in executed code."""
        executor = SafeCodeExecutor()
        
        invalid_code = "def incomplete_function("
        
        result = executor.execute_code(invalid_code)
        assert not result["success"]
        assert "security validation failed" in result["error"].lower()


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_hash_code(self):
        """Test code hashing function."""
        code1 = "def hello(): print('hello')"
        code2 = "def hello(): print('hello')"
        code3 = "def goodbye(): print('goodbye')"
        
        hash1 = hash_code(code1)
        hash2 = hash_code(code2)
        hash3 = hash_code(code3)
        
        # Same code should produce same hash
        assert hash1 == hash2
        
        # Different code should produce different hash
        assert hash1 != hash3
        
        # Hashes should be strings
        assert isinstance(hash1, str)

    def test_sanitize_code_for_display(self):
        """Test code sanitization for display."""
        sensitive_code = """
password = "secret123"
api_key = 'my-secret-key'
token = "bearer-token"
def process_data():
    return "processed"
"""
        
        sanitized = sanitize_code_for_display(sensitive_code)
        
        # Sensitive values should be masked
        assert "secret123" not in sanitized
        assert "my-secret-key" not in sanitized
        assert "bearer-token" not in sanitized
        assert '***' in sanitized
        
        # Function should still be visible
        assert "def process_data" in sanitized

    def test_sanitize_long_code(self):
        """Test sanitization of long code."""
        long_code = "x = 1\n" * 1000  # 1000 lines
        
        sanitized = sanitize_code_for_display(long_code, max_length=100)
        
        assert len(sanitized) <= 103  # 100 + "..."
        assert sanitized.endswith("...")

    def test_sanitize_short_code(self):
        """Test sanitization of short code (no truncation)."""
        short_code = "def add(a, b): return a + b"
        
        sanitized = sanitize_code_for_display(short_code, max_length=100)
        
        assert sanitized == short_code  # Should be unchanged
