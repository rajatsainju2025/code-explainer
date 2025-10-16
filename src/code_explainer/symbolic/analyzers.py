"""Analysis methods for symbolic analysis."""

import ast
from typing import Any, Dict, List


class ComplexityAnalyzers:
    """Methods for analyzing code complexity and data flow."""

    def __init__(self):
        self.variable_assignments: Dict[str, List[ast.AST]] = {}

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

    def _get_variable_dependencies(self, expr: ast.AST) -> List[str]:
        """Get variables that this expression depends on."""
        dependencies = []
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                dependencies.append(node.id)
        return dependencies