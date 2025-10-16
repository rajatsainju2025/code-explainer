"""Property test generation methods for symbolic analysis."""

import ast
from typing import List

from .models import PropertyTest


class PropertyGenerators:
    """Methods for generating property-based tests."""

    def _generate_property_tests(self, tree: ast.AST, code: str) -> List[PropertyTest]:
        """Generate property-based tests for the code."""
        tests = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                tests.extend(self._generate_function_property_tests(node, code))

        return tests

    def _generate_function_property_tests(
        self, node: ast.FunctionDef, code: str
    ) -> List[PropertyTest]:
        """Generate property-based tests for a function."""
        tests = []

        # Basic property: function should not crash with valid inputs
        test_name = f"test_{node.name}_no_crash"
        property_desc = f"Function {node.name} should not crash with valid inputs"

        # Generate basic test code
        args = [arg.arg for arg in node.args.args]
        test_code = f"""
def {test_name}():
    # Property: {property_desc}
    try:
        result = {node.name}({', '.join(f'valid_{arg}' for arg in args)})
        assert result is not None or result == 0 or result == [] or result == ""
        return True
    except Exception as e:
        return False, str(e)
"""

        tests.append(
            PropertyTest(
                test_name=test_name,
                property_description=property_desc,
                test_code=test_code.strip(),
                input_constraints=[f"{arg}: valid input" for arg in args],
                expected_behavior="Function executes without errors",
            )
        )

        # Add specific property tests based on function analysis
        if self._appears_to_be_sorting_function(node):
            tests.append(self._generate_sorting_property_test(node))
        elif self._appears_to_be_math_function(node):
            tests.append(self._generate_math_property_test(node))

        return tests

    def _appears_to_be_sorting_function(self, node: ast.FunctionDef) -> bool:
        """Check if function appears to be sorting-related."""
        keywords = ["sort", "order", "rank"]
        return any(kw in node.name.lower() for kw in keywords)

    def _appears_to_be_math_function(self, node: ast.FunctionDef) -> bool:
        """Check if function appears to be math-related."""
        math_ops = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow]
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.BinOp) and type(stmt.op) in math_ops:
                return True
        return False

    def _generate_sorting_property_test(self, node: ast.FunctionDef) -> PropertyTest:
        """Generate property test for sorting function."""
        test_name = f"test_{node.name}_sorting_property"
        return PropertyTest(
            test_name=test_name,
            property_description="Sorted output should be permutation of input",
            test_code=f"""
def {test_name}():
    import random
    test_input = random.sample(range(100), 10)
    result = {node.name}(test_input.copy())
    return sorted(test_input) == sorted(result)
""".strip(),
            input_constraints=["list of comparable elements"],
            expected_behavior="Output is sorted permutation of input",
        )

    def _generate_math_property_test(self, node: ast.FunctionDef) -> PropertyTest:
        """Generate property test for math function."""
        test_name = f"test_{node.name}_math_property"
        return PropertyTest(
            test_name=test_name,
            property_description="Math function should preserve basic properties",
            test_code=f"""
def {test_name}():
    # Test with known values
    test_cases = [(0, 1, 2), (1, 1, 1), (-1, 0, 1)]
    for inputs in test_cases:
        try:
            result = {node.name}(*inputs)
            assert isinstance(result, (int, float, complex))
        except Exception:
            return False
    return True
""".strip(),
            input_constraints=["numeric inputs"],
            expected_behavior="Returns numeric result for valid inputs",
        )