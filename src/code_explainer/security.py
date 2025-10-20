"""
Security utilities for code validation and safe execution.
"""

from typing import Dict, Any, List, Tuple
import ast
import re


class CodeSecurityValidator:
    """Validates code for security issues."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'os\.system',
            r'subprocess\.',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'open\s*\(',
            r'file\s*\(',
            r'import\s+requests',
            r'import\s+urllib',
            r'urllib\.',
            r'requests\.',
            r'socket\.',
            r'import\s+socket'
        ]

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate code for security issues."""
        issues = []

        # Check for dangerous imports and functions
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code):
                if 'open' in pattern:
                    issues.append("Potentially dangerous file operation detected")
                elif 'import' in pattern and ('os' in pattern or 'subprocess' in pattern or 'sys' in pattern):
                    issues.append("Potentially dangerous system import detected")
                elif 'eval' in pattern or 'exec' in pattern:
                    issues.append("Potentially dangerous code execution detected")
                elif 'requests' in pattern or 'urllib' in pattern or 'socket' in pattern:
                    issues.append("Potentially dangerous network operation detected")
                else:
                    issues.append(f"Potentially dangerous pattern detected: {pattern}")

        # Parse AST to check for other issues
        try:
            tree = ast.parse(code)
            issues.extend(self._analyze_ast(tree))
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")

        return len(issues) == 0, issues

    def _analyze_ast(self, tree: ast.AST) -> List[str]:
        """Analyze AST for security issues."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'sys']:
                        issues.append(f"Dangerous import: {alias.name}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'open']:
                        issues.append(f"Dangerous function call: {node.func.id}")

        return issues


class SafeCodeExecutor:
    """Executes code in a safe environment."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code safely."""
        # First validate the code
        validator = CodeSecurityValidator()
        is_safe, issues = validator.validate_code(code)

        if not is_safe:
            return {
                "success": False,
                "error": "Security validation failed",
                "issues": issues
            }

        import subprocess
        import tempfile
        import os

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Clean up
            os.unlink(temp_file)

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "execution_time": 0.1  # Simplified
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Timeout: execution timed out after {self.timeout} seconds",
                "output": "",
                "execution_time": self.timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": 0.1
            }


def hash_code(code: str) -> str:
    """Generate a hash for code content."""
    import hashlib
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


def sanitize_code_for_display(code: str, max_length: int = 1000) -> str:
    """Sanitize code for safe display."""
    import re

    # Remove potentially dangerous patterns for display
    sanitized = code.replace('import os', '# import os (removed for security)')
    sanitized = sanitized.replace('import subprocess', '# import subprocess (removed for security)')
    sanitized = sanitized.replace('eval(', '# eval( (removed for security)')
    sanitized = sanitized.replace('exec(', '# exec( (removed for security)')

    # Mask potential secrets (simple pattern matching)
    sanitized = re.sub(r'password\s*=\s*["\'][^"\']*["\']', 'password = "***"', sanitized)
    sanitized = re.sub(r'api_key\s*=\s*["\'][^"\']*["\']', 'api_key = "***"', sanitized)
    sanitized = re.sub(r'token\s*=\s*["\'][^"\']*["\']', 'token = "***"', sanitized)
    sanitized = re.sub(r'secret\s*=\s*["\'][^"\']*["\']', 'secret = "***"', sanitized)

    # Truncate if too long
    if len(sanitized) > max_length:
        # Truncate at max_length - len("...") to ensure total length <= max_length + 3
        truncate_at = max_length - 3
        sanitized = sanitized[:truncate_at] + "..."

    return sanitized