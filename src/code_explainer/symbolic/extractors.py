"""Condition extraction methods for symbolic analysis."""

import ast
from typing import List, Optional, Set

from .models import SymbolicCondition, PropertyTest


class ConditionExtractors:
    """Methods for extracting various types of conditions from AST."""

    def __init__(self):
        self.variable_assignments: dict = {}
        self.function_calls: List[ast.Call] = []

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