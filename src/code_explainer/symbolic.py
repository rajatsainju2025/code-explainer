"""Symbolic explanation generation for code analysis."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SymbolicCondition:
    """Represents a symbolic condition in code."""

    condition_type: str  # 'input', 'precondition', 'postcondition', 'invariant'
    expression: str
    line_number: int
    confidence: float
    variables: Set[str]


@dataclass
class PropertyTest:
    """Represents a property-based test."""

    test_name: str
    property_description: str
    test_code: str
    input_constraints: List[str]
    expected_behavior: str


@dataclass
class SymbolicExplanation:
    """Complete symbolic explanation of code."""

    input_conditions: List[SymbolicCondition]
    preconditions: List[SymbolicCondition]
    postconditions: List[SymbolicCondition]
    invariants: List[SymbolicCondition]
    property_tests: List[PropertyTest]
    complexity_analysis: Dict[str, Any]
    data_flow: Dict[str, List[str]]


class SymbolicAnalyzer:
    """Analyzes code to extract symbolic conditions and generate property tests."""

    def __init__(self):
        self.variable_assignments: Dict[str, List[ast.AST]] = {}
        self.function_calls: List[ast.Call] = []
        self.control_flow: List[ast.AST] = []

    def analyze_code(self, code: str) -> SymbolicExplanation:
        """Generate symbolic explanation for given code."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # Return empty explanation for invalid code
            return SymbolicExplanation([], [], [], [], [], {}, {})

        self._reset_state()
        self._analyze_ast(tree)

        return SymbolicExplanation(
            input_conditions=self._extract_input_conditions(tree),
            preconditions=self._extract_preconditions(tree),
            postconditions=self._extract_postconditions(tree),
            invariants=self._extract_invariants(tree),
            property_tests=self._generate_property_tests(tree, code),
            complexity_analysis=self._analyze_complexity(tree),
            data_flow=self._analyze_data_flow(tree),
        )

    def _reset_state(self):
        """Reset analyzer state for new code."""
        self.variable_assignments.clear()
        self.function_calls.clear()
        self.control_flow.clear()

    def _analyze_ast(self, tree: ast.AST):
        """Walk AST and collect analysis data."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.variable_assignments.setdefault(target.id, []).append(node)
            elif isinstance(node, ast.Call):
                self.function_calls.append(node)
            elif isinstance(node, (ast.If, ast.While, ast.For)):
                self.control_flow.append(node)

    def _extract_input_conditions(self, tree: ast.AST) -> List[SymbolicCondition]:
        """Extract input validation conditions."""
        conditions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Look for input validation patterns
                for stmt in node.body[:3]:  # Check first few statements
                    if isinstance(stmt, ast.If):
                        condition = self._extract_condition_from_if(stmt, "input")
                        if condition:
                            conditions.append(condition)
                    elif isinstance(stmt, ast.Assert):
                        condition = self._extract_condition_from_assert(stmt, "input")
                        if condition:
                            conditions.append(condition)

        return conditions

    def _extract_preconditions(self, tree: ast.AST) -> List[SymbolicCondition]:
        """Extract preconditions from code."""
        conditions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                condition = self._extract_condition_from_assert(node, "precondition")
                if condition:
                    conditions.append(condition)
            elif isinstance(node, ast.If) and self._is_guard_condition(node):
                condition = self._extract_condition_from_if(node, "precondition")
                if condition:
                    conditions.append(condition)

        return conditions

    def _extract_postconditions(self, tree: ast.AST) -> List[SymbolicCondition]:
        """Extract postconditions from return statements and end of functions."""
        conditions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check return statements
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Return) and stmt.value:
                        condition = self._analyze_return_condition(stmt)
                        if condition:
                            conditions.append(condition)

        return conditions

    def _extract_invariants(self, tree: ast.AST) -> List[SymbolicCondition]:
        """Extract loop invariants and class invariants."""
        conditions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.While, ast.For)):
                invariant = self._analyze_loop_invariant(node)
                if invariant:
                    conditions.append(invariant)

        return conditions

    def _generate_property_tests(self, tree: ast.AST, code: str) -> List[PropertyTest]:
        """Generate property-based tests for the code."""
        tests = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                tests.extend(self._generate_function_property_tests(node, code))

        return tests

    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze computational complexity."""
        complexity: Dict[str, Any] = {
            "cyclomatic_complexity": self._calculate_cyclomatic_complexity(tree),
            "nesting_depth": self._calculate_nesting_depth(tree),
            "number_of_loops": len(
                [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
            ),
            "number_of_conditions": len([n for n in ast.walk(tree) if isinstance(n, ast.If)]),
        }

        # Estimate time complexity
        complexity["estimated_time_complexity"] = self._estimate_time_complexity(tree)

        return complexity

    def _analyze_data_flow(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze data flow between variables."""
        data_flow = {}

        for var_name, assignments in self.variable_assignments.items():
            dependencies = []
            for assign_node in assignments:
                if isinstance(assign_node, ast.Assign):
                    deps = self._get_variable_dependencies(assign_node.value)
                    dependencies.extend(deps)
            data_flow[var_name] = list(set(dependencies))

        return data_flow

    def _extract_condition_from_if(
        self, node: ast.If, condition_type: str
    ) -> Optional[SymbolicCondition]:
        """Extract symbolic condition from if statement."""
        try:
            condition_text = ast.unparse(node.test)
            variables = self._extract_variables_from_expr(node.test)

            return SymbolicCondition(
                condition_type=condition_type,
                expression=condition_text,
                line_number=getattr(node, "lineno", 0),
                confidence=0.8,
                variables=variables,
            )
        except Exception:
            return None

    def _extract_condition_from_assert(
        self, node: ast.Assert, condition_type: str
    ) -> Optional[SymbolicCondition]:
        """Extract symbolic condition from assert statement."""
        try:
            condition_text = ast.unparse(node.test)
            variables = self._extract_variables_from_expr(node.test)

            return SymbolicCondition(
                condition_type=condition_type,
                expression=condition_text,
                line_number=getattr(node, "lineno", 0),
                confidence=0.9,  # Higher confidence for explicit assertions
                variables=variables,
            )
        except Exception:
            return None

    def _is_guard_condition(self, node: ast.If) -> bool:
        """Check if if statement is a guard condition."""
        # Simple heuristic: if it raises an exception or returns early
        for stmt in node.body:
            if isinstance(stmt, (ast.Raise, ast.Return)):
                return True
        return False

    def _analyze_return_condition(self, node: ast.Return) -> Optional[SymbolicCondition]:
        """Analyze return statement to extract postcondition."""
        if not node.value:
            return None

        try:
            return_expr = ast.unparse(node.value)
            variables = self._extract_variables_from_expr(node.value)

            return SymbolicCondition(
                condition_type="postcondition",
                expression=f"returns {return_expr}",
                line_number=getattr(node, "lineno", 0),
                confidence=0.7,
                variables=variables,
            )
        except Exception:
            return None

    def _analyze_loop_invariant(self, node: ast.AST) -> Optional[SymbolicCondition]:
        """Analyze loop to extract invariant."""
        # Simple heuristic: look for variables that maintain relationships
        try:
            if isinstance(node, ast.While):
                condition_text = ast.unparse(node.test)
                variables = self._extract_variables_from_expr(node.test)

                return SymbolicCondition(
                    condition_type="invariant",
                    expression=f"loop_condition: {condition_text}",
                    line_number=getattr(node, "lineno", 0),
                    confidence=0.6,
                    variables=variables,
                )
        except Exception:
            pass
        return None

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
            property_description=f"Sorted output should be permutation of input",
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
            property_description=f"Math function should preserve basic properties",
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

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""

        def get_depth(node, current_depth=0):
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth

        return get_depth(tree)

    def _estimate_time_complexity(self, tree: ast.AST) -> str:
        """Estimate time complexity based on loop nesting."""
        max_loop_nesting = 0

        def count_loop_nesting(node, depth=0):
            nonlocal max_loop_nesting
            if isinstance(node, (ast.For, ast.While)):
                depth += 1
                max_loop_nesting = max(max_loop_nesting, depth)

            for child in ast.iter_child_nodes(node):
                count_loop_nesting(child, depth)

        count_loop_nesting(tree)

        if max_loop_nesting == 0:
            return "O(1)"
        elif max_loop_nesting == 1:
            return "O(n)"
        elif max_loop_nesting == 2:
            return "O(n²)"
        elif max_loop_nesting == 3:
            return "O(n³)"
        else:
            return f"O(n^{max_loop_nesting})"

    def _extract_variables_from_expr(self, expr: ast.AST) -> Set[str]:
        """Extract variable names from expression."""
        variables = set()
        for node in ast.walk(expr):
            if isinstance(node, ast.Name):
                variables.add(node.id)
        return variables

    def _get_variable_dependencies(self, expr: ast.AST) -> List[str]:
        """Get variables that this expression depends on."""
        dependencies = []
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                dependencies.append(node.id)
        return dependencies


def format_symbolic_explanation(explanation: SymbolicExplanation) -> str:
    """Format symbolic explanation as human-readable text."""
    sections = []

    if explanation.input_conditions:
        sections.append("Input Conditions:")
        for cond in explanation.input_conditions:
            sections.append(f"  - {cond.expression} (confidence: {cond.confidence:.1f})")

    if explanation.preconditions:
        sections.append("\nPreconditions:")
        for cond in explanation.preconditions:
            sections.append(f"  - {cond.expression} (confidence: {cond.confidence:.1f})")

    if explanation.postconditions:
        sections.append("\nPostconditions:")
        for cond in explanation.postconditions:
            sections.append(f"  - {cond.expression} (confidence: {cond.confidence:.1f})")

    if explanation.invariants:
        sections.append("\nInvariants:")
        for cond in explanation.invariants:
            sections.append(f"  - {cond.expression} (confidence: {cond.confidence:.1f})")

    if explanation.complexity_analysis:
        sections.append("\nComplexity Analysis:")
        comp = explanation.complexity_analysis
        sections.append(f"  - Time Complexity: {comp.get('estimated_time_complexity', 'Unknown')}")
        sections.append(f"  - Cyclomatic Complexity: {comp.get('cyclomatic_complexity', 0)}")
        sections.append(f"  - Max Nesting Depth: {comp.get('nesting_depth', 0)}")

    if explanation.property_tests:
        sections.append("\nProperty-Based Tests:")
        for test in explanation.property_tests[:3]:  # Show first 3 tests
            sections.append(f"  - {test.property_description}")

    return "\n".join(sections) if sections else "No symbolic conditions detected."
