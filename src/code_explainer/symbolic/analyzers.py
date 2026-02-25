"""Analysis methods for symbolic analysis."""

import ast
from typing import Any, Dict, List, Tuple

# Pre-compile the tuple of nesting-depth node types once at module load.
# Re-building the same literal tuple inside a recursive inner function
# (get_depth) on every recursive call is unnecessary allocation.
_NESTING_TYPES = (ast.If, ast.While, ast.For, ast.With, ast.Try)


class ComplexityAnalyzers:
    """Methods for analyzing code complexity and data flow."""

    def __init__(self):
        self.variable_assignments: Dict[str, List[ast.AST]] = {}

    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze computational complexity with optimized single-pass traversal."""
        # Single-pass traversal to count loops, conditions, and calculate metrics
        loop_count, condition_count, max_loop_nesting = self._count_nodes_single_pass(tree)
        
        complexity: Dict[str, Any] = {
            "cyclomatic_complexity": self._calculate_cyclomatic_complexity(tree),
            "nesting_depth": self._calculate_nesting_depth(tree),
            "number_of_loops": loop_count,
            "number_of_conditions": condition_count,
        }

        # Estimate time complexity based on pre-calculated nesting
        complexity["estimated_time_complexity"] = self._estimate_complexity_from_nesting(
            max_loop_nesting
        )

        return complexity

    def _count_nodes_single_pass(self, tree: ast.AST) -> Tuple[int, int, int]:
        """Single-pass traversal to count loops, conditions, and max loop nesting.
        
        Returns:
            Tuple of (loop_count, condition_count, max_loop_nesting)
        """
        loop_count = 0
        condition_count = 0
        max_loop_nesting = 0
        
        def traverse(node, current_loop_depth=0):
            nonlocal loop_count, condition_count, max_loop_nesting
            
            if isinstance(node, (ast.For, ast.While)):
                loop_count += 1
                current_loop_depth += 1
                max_loop_nesting = max(max_loop_nesting, current_loop_depth)
            elif isinstance(node, ast.If):
                condition_count += 1
            
            for child in ast.iter_child_nodes(node):
                traverse(child, current_loop_depth)
        
        traverse(tree)
        return loop_count, condition_count, max_loop_nesting

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
                if isinstance(child, _NESTING_TYPES):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth

        return get_depth(tree)


    @staticmethod
    def _estimate_complexity_from_nesting(max_loop_nesting: int) -> str:
        """Convert loop nesting depth to Big-O notation."""
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
        # dict.fromkeys preserves insertion order while deduplicating in O(n),
        # avoiding the list→set→list round-trip of list(set(dependencies)).
        return list(dict.fromkeys(dependencies))