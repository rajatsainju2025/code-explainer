"""Test coverage improvements and testing utilities.

Provides utilities and patterns for improving test coverage and quality.
"""

import unittest
from typing import Any, Callable, Optional


class MockCodeExplainer:
    """Mock CodeExplainer for testing."""
    
    def __init__(self):
        self.explain_code_called = False
        self.last_code = None
        self.last_strategy = None
    
    def explain_code(self, code: str, strategy: Optional[str] = None) -> str:
        """Mock explain_code method."""
        self.explain_code_called = True
        self.last_code = code
        self.last_strategy = strategy
        return f"Mock explanation of: {code[:50]}"


class BaseCodeExplainerTest(unittest.TestCase):
    """Base test class with common setup."""
    
    def setUp(self) -> None:
        """Setup test fixtures."""
        self.test_code = """
def hello():
    print("Hello, World!")
        """
        self.test_strategies = [
            "vanilla",
            "ast_augmented",
            "retrieval_augmented",
            "execution_trace"
        ]
    
    def assert_valid_explanation(self, explanation: str) -> None:
        """Assert explanation is valid.
        
        Args:
            explanation: The explanation to validate
        """
        self.assertIsInstance(explanation, str)
        self.assertTrue(len(explanation) > 0)
        self.assertTrue(len(explanation) < 100000)
    
    def assert_code_valid(self, code: str) -> None:
        """Assert code is valid Python.
        
        Args:
            code: The code to validate
        """
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError:
            self.fail(f"Invalid Python syntax: {code}")


# Common test patterns
COVERAGE_PRIORITIES = {
    "HIGH": [
        "model/core.py - CodeExplainer main class",
        "validation.py - Input validation",
        "security.py - Security checks",
    ],
    "MEDIUM": [
        "retrieval/ - Search functionality",
        "cache/ - Caching mechanisms",
        "multi_agent/ - Agent orchestration",
    ],
    "LOW": [
        "symbolic/ - Symbolic analysis",
        "utils/ - Utility functions",
        "api/ - API endpoints",
    ]
}
