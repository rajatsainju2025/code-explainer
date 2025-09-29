"""Security utilities for code execution and validation."""

import ast
import hashlib
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CodeSecurityValidator:
    """Validates code for security risks before execution or analysis."""

    # Dangerous imports and functions
    DANGEROUS_IMPORTS = {
        "os", "subprocess", "sys", "shutil", "socket", "urllib", "requests",
        "pickle", "marshal", "imp", "importlib", "__import__", "eval", "exec",
        "compile", "globals", "locals", "vars", "dir", "getattr", "setattr",
        "delattr", "hasattr", "open", "file", "input", "raw_input"
    }

    DANGEROUS_FUNCTIONS = {
        "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
        "globals", "locals", "vars", "dir", "open", "file", "input", "raw_input"
    }

    DANGEROUS_MODULES = {
        "os.system", "os.popen", "os.spawn", "subprocess.call", "subprocess.run",
        "subprocess.Popen", "socket.socket", "pickle.loads", "marshal.loads"
    }

    def __init__(self, strict_mode: bool = True):
        """Initialize the security validator.

        Args:
            strict_mode: If True, applies stricter security checks
        """
        self.strict_mode = strict_mode

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate code for security risks.

        Args:
            code: The code to validate

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        try:
            # Parse the code to AST
            tree = ast.parse(code)

            # Check for dangerous patterns
            issues.extend(self._check_imports(tree))
            issues.extend(self._check_function_calls(tree))
            issues.extend(self._check_string_patterns(code))

            if self.strict_mode:
                issues.extend(self._check_strict_patterns(tree, code))

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"Code validation error: {e}")

        return len(issues) == 0, issues

    def _check_imports(self, tree: ast.AST) -> List[str]:
        """Check for dangerous imports."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.DANGEROUS_IMPORTS:
                        issues.append(f"Dangerous import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module in self.DANGEROUS_IMPORTS:
                    issues.append(f"Dangerous import from: {node.module}")

                for alias in node.names:
                    if alias.name in self.DANGEROUS_FUNCTIONS:
                        issues.append(f"Dangerous function import: {alias.name}")

        return issues

    def _check_function_calls(self, tree: ast.AST) -> List[str]:
        """Check for dangerous function calls."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                if func_name in self.DANGEROUS_FUNCTIONS:
                    issues.append(f"Dangerous function call: {func_name}")
                elif func_name in self.DANGEROUS_MODULES:
                    issues.append(f"Dangerous module function: {func_name}")

        return issues

    def _check_string_patterns(self, code: str) -> List[str]:
        """Check for dangerous string patterns."""
        issues = []

        # Check for shell commands
        shell_patterns = [
            r'os\.system\s*\(',
            r'subprocess\.',
            r'Popen\s*\(',
            r'call\s*\(',
            r'check_output\s*\(',
        ]

        for pattern in shell_patterns:
            if re.search(pattern, code):
                issues.append(f"Potential shell execution: {pattern}")

        # Check for file operations
        file_patterns = [
            r'open\s*\(',
            r'file\s*\(',
            r'with\s+open',
        ]

        for pattern in file_patterns:
            if re.search(pattern, code):
                issues.append(f"File operation detected: {pattern}")

        # Check for network operations
        network_patterns = [
            r'socket\.',
            r'urllib\.',
            r'requests\.',
            r'http\.',
        ]

        for pattern in network_patterns:
            if re.search(pattern, code):
                issues.append(f"Network operation detected: {pattern}")

        return issues

    def _check_strict_patterns(self, tree: ast.AST, code: str) -> List[str]:
        """Additional strict mode checks."""
        issues = []

        # Check for attribute access to dangerous modules
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id in self.DANGEROUS_IMPORTS:
                        issues.append(f"Attribute access to dangerous module: {node.value.id}.{node.attr}")

        # Check for dynamic code execution patterns
        dynamic_patterns = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'compile\s*\(',
            r'__import__\s*\(',
        ]

        for pattern in dynamic_patterns:
            if re.search(pattern, code):
                issues.append(f"Dynamic code execution: {pattern}")

        return issues

    def _get_function_name(self, func_node: ast.AST) -> str:
        """Extract function name from AST node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            if isinstance(func_node.value, ast.Name):
                return f"{func_node.value.id}.{func_node.attr}"
        return ""


class SafeCodeExecutor:
    """Safely execute code with restrictions and timeouts."""

    def __init__(self, timeout: int = 10, max_memory_mb: int = 100):
        """Initialize the safe executor.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.validator = CodeSecurityValidator()

    def execute_code(self, code: str, capture_output: bool = True) -> Dict[str, Any]:
        """Safely execute code and return results.

        Args:
            code: The code to execute
            capture_output: Whether to capture stdout/stderr

        Returns:
            Dictionary with execution results
        """
        # Validate code first
        is_safe, issues = self.validator.validate_code(code)
        if not is_safe:
            return {
                "success": False,
                "error": f"Security validation failed: {'; '.join(issues)}",
                "output": "",
                "stderr": ""
            }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute in subprocess; rely on Python's timeout for portability
            cmd = [
                "python", temp_file
            ]
            if capture_output:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            else:
                result = subprocess.run(cmd, timeout=self.timeout)
                return {
                    "success": result.returncode == 0,
                    "return_code": result.returncode
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timeout ({self.timeout}s)",
                "output": "",
                "stderr": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {e}",
                "output": "",
                "stderr": ""
            }
        finally:
            # Clean up temporary file
            try:
                Path(temp_file).unlink()
            except Exception:
                pass


def hash_code(code: str) -> str:
    """Generate a hash for code content.

    Args:
        code: The code to hash

    Returns:
        SHA256 hash of the code
    """
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


def sanitize_code_for_display(code: str, max_length: int = 1000) -> str:
    """Sanitize code for safe display in logs or UI.

    Args:
        code: The code to sanitize
        max_length: Maximum length of displayed code

    Returns:
        Sanitized code string
    """
    # Remove sensitive patterns
    sanitized = re.sub(r'(password|token|key|secret)\s*=\s*["\'][^"\']*["\']',
                      r'\1="***"', code, flags=re.IGNORECASE)

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized
