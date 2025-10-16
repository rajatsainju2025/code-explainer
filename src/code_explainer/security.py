"""
Security utilities for code validation and safe execution.
"""

from typing import Dict, Any, List, Tuple
import ast
import re


class CodeSecurityValidator:
    """Validates code for security issues."""

    def __init__(self):
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
            r'file\s*\('
        ]

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate code for security issues."""
        issues = []

        # Check for dangerous imports and functions
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code):
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
    """Executes code safely with resource limits."""

    def __init__(self, timeout: int = 10, max_memory_mb: int = 100):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code safely."""
        # Placeholder - in real implementation would use restricted execution
        try:
            # Basic validation first
            validator = CodeSecurityValidator()
            is_safe, issues = validator.validate_code(code)

            if not is_safe:
                return {
                    "success": False,
                    "error": "Security validation failed",
                    "issues": issues
                }

            # For now, just return success without actual execution
            return {
                "success": True,
                "output": "Code validated successfully",
                "execution_time": 0.1
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }