"""Condition extraction methods for symbolic analysis."""

import ast
from typing import FrozenSet, List, Optional

from .models import SymbolicCondition

# Pre-cache AST types for O(1) lookup
_AST_FUNCTIONDEF = ast.FunctionDef
_AST_IF = ast.If
_AST_ASSERT = ast.Assert
_AST_RETURN = ast.Return
_AST_WHILE = ast.While
_AST_FOR = ast.For
_AST_NAME = ast.Name
_AST_RAISE = ast.Raise
_AST_LOAD = ast.Load
_AST_LOOP_TYPES = (_AST_WHILE, _AST_FOR)
_AST_EARLY_EXIT = (_AST_RAISE, _AST_RETURN)



class ConditionExtractors:
    """Methods for extracting various types of conditions from AST."""

    __slots__ = ("variable_assignments",)

    def __init__(self):
        self.variable_assignments: dict = {}

    def _extract_input_conditions(self, tree: ast.AST) -> List[SymbolicCondition]:
        """Extract input validation conditions."""
        conditions = []

        for node in ast.walk(tree):
            if type(node) is _AST_FUNCTIONDEF:
                # Look for input validation patterns in first 3 statements
                for stmt in node.body[:3]:
                    stmt_type = type(stmt)
                    if stmt_type is _AST_IF:
                        condition = self._extract_condition_from_if(stmt, "input")
                        if condition:
                            conditions.append(condition)
                    elif stmt_type is _AST_ASSERT:
                        condition = self._extract_condition_from_assert(stmt, "input")
                        if condition:
                            conditions.append(condition)

        return conditions

    def _extract_preconditions(self, tree: ast.AST) -> List[SymbolicCondition]:
        """Extract preconditions from code."""
        conditions = []

        for node in ast.walk(tree):
            node_type = type(node)
            if node_type is _AST_ASSERT:
                condition = self._extract_condition_from_assert(node, "precondition")
                if condition:
                    conditions.append(condition)
            elif node_type is _AST_IF and self._is_guard_condition(node):
                condition = self._extract_condition_from_if(node, "precondition")
                if condition:
                    conditions.append(condition)

        return conditions

    def _extract_postconditions(self, tree: ast.AST) -> List[SymbolicCondition]:
        """Extract postconditions from return statements and end of functions.

        A single ast.walk(tree) is sufficient: Return statements can only
        appear inside functions (enforced by the parser), so filtering to
        nodes inside FunctionDef subtrees via a nested walk is redundant.
        The previous nested-walk approach visited Return nodes multiple times
        for code containing nested function definitions.
        """
        conditions = []

        for node in ast.walk(tree):
            if type(node) is _AST_RETURN and node.value:
                condition = self._analyze_return_condition(node)
                if condition:
                    conditions.append(condition)

        return conditions

    def _extract_invariants(self, tree: ast.AST) -> List[SymbolicCondition]:
        """Extract loop invariants and class invariants."""
        conditions = []

        for node in ast.walk(tree):
            if isinstance(node, _AST_LOOP_TYPES):
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
            if isinstance(stmt, _AST_EARLY_EXIT):
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
            if type(node) is _AST_WHILE:
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

    def _extract_variables_from_expr(self, expr: ast.AST) -> FrozenSet[str]:
        """Extract variable names from expression (returns immutable frozenset)."""
        return frozenset(node.id for node in ast.walk(expr) if type(node) is _AST_NAME)

    def _get_variable_dependencies(self, expr: ast.AST) -> List[str]:
        """Get variables that this expression depends on."""
        return [
            node.id
            for node in ast.walk(expr)
            if type(node) is _AST_NAME and type(node.ctx) is _AST_LOAD
        ]