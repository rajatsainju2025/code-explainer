"""
Code quality analysis utilities with optimized performance.

Performance optimizations:
- Pre-compiled regex patterns at module level
- LRU caching for AST parsing
- __slots__ for memory efficiency
- Frozen node type sets for O(1) lookups
- Single-pass scoring calculations
"""

from functools import lru_cache
from typing import Dict, Any, List, Tuple, FrozenSet
import ast
import re
from enum import Enum

# Pre-compile regex patterns at module level for efficiency
SNAKE_CASE_PATTERN = re.compile(r'^[a-z_][a-z0-9_]*$')
PASCAL_CASE_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*$')

# Pre-frozen type sets for O(1) lookups
_STATEMENT_TYPES: FrozenSet[type] = frozenset({
    ast.Assign, ast.Return, ast.If, ast.For, ast.While
})

# Severity weights for fast lookup
_SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 1.0,
    "high": 0.7,
    "medium": 0.4,
    "low": 0.1
}


class IssueLevel(Enum):
    """Severity levels for code quality issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityIssue:
    """Represents a code quality issue."""
    
    __slots__ = ('level', 'message', 'rule_id', 'line', 'column')

    def __init__(self, level, message: str, rule_id: str, line: int, column: int = 0):
        self.level = level
        self.message = message
        self.rule_id = rule_id
        self.line = line
        self.column = column


class CodeQualityAnalyzer:
    """Analyzes code quality and provides suggestions."""
    
    __slots__ = ('_ast_cache',)

    def __init__(self):
        # Cache for parsed ASTs to avoid reparsing
        self._ast_cache: Dict[str, ast.AST] = {}

    @lru_cache(maxsize=256)
    def _parse_code_cached(self, code: str) -> Tuple[bool, Any]:
        """Parse code and cache result. Returns (success, tree_or_error)."""
        try:
            tree = ast.parse(code)
            return (True, tree)
        except SyntaxError as e:
            return (False, e)

    def analyze_code(self, code: str) -> List[QualityIssue]:
        """Analyze code quality - main method expected by tests."""
        return self.analyze_quality(code)["issues"]

    def analyze_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality."""
        issues = []
        suggestions = []

        # Use cached parsing
        success, result = self._parse_code_cached(code)
        
        if not success:
            e = result
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                message=f"Syntax error: {e.msg}",
                rule_id="syntax_error",
                line=e.lineno or 1,
                column=e.offset or 0
            ))
            suggestions.append("Fix syntax errors before analysis")
        else:
            tree = result
            # Check for various quality issues
            issues.extend(self._check_naming_conventions(tree))
            issues.extend(self._check_complexity(tree))
            issues.extend(self._check_best_practices(tree))

            # Generate suggestions
            suggestions = self._generate_suggestions(issues, code)

        # Calculate scores
        complexity_score = self._calculate_complexity_score(issues)
        readability_score = self._calculate_readability_score(code, issues)
        maintainability_score = self._calculate_maintainability_score(issues)

        return {
            "complexity_score": complexity_score,
            "readability_score": readability_score,
            "maintainability_score": maintainability_score,
            "issues": issues,
            "suggestions": suggestions
        }

    def _check_naming_conventions(self, tree: ast.AST) -> List[QualityIssue]:
        """Check naming conventions using pre-compiled patterns."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not SNAKE_CASE_PATTERN.match(node.name):
                    issues.append(QualityIssue(
                        level=IssueLevel.MEDIUM,
                        message=f"Function '{node.name}' should use snake_case",
                        rule_id="naming_convention",
                        line=node.lineno,
                        column=node.col_offset
                    ))
            elif isinstance(node, ast.ClassDef):
                if not PASCAL_CASE_PATTERN.match(node.name):
                    issues.append(QualityIssue(
                        level=IssueLevel.MEDIUM,
                        message=f"Class '{node.name}' should use PascalCase",
                        rule_id="naming_convention",
                        line=node.lineno,
                        column=node.col_offset
                    ))

        return issues

    def _check_complexity(self, tree: ast.AST) -> List[QualityIssue]:
        """Check code complexity using frozen type sets for O(1) lookups."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count statements using frozen set for O(1) type checking
                statements = sum(1 for n in ast.walk(node) if type(n) in _STATEMENT_TYPES)
                if statements > 20:
                    issues.append(QualityIssue(
                        level=IssueLevel.MEDIUM,
                        message=f"Function '{node.name}' is too complex ({statements} statements)",
                        rule_id="complexity",
                        line=node.lineno,
                        column=node.col_offset
                    ))

        return issues

    def _check_best_practices(self, tree: ast.AST) -> List[QualityIssue]:
        """Check best practices."""
        issues = []

        # Check for eval usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'eval':
                issues.append(QualityIssue(
                    level=IssueLevel.HIGH,
                    message="Use of eval() is dangerous and should be avoided",
                    rule_id="dangerous_eval",
                    line=node.lineno,
                    column=node.col_offset
                ))

        return issues

    def _generate_suggestions(self, issues: List[QualityIssue], code: str) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        if not issues:
            suggestions.append("Code follows good practices")
            return suggestions

        # Group issues by type
        issue_counts = {}
        for issue in issues:
            level = issue.level.value
            issue_counts[level] = issue_counts.get(level, 0) + 1

        if issue_counts.get("critical", 0) > 0:
            suggestions.append("Fix critical issues first")
        if issue_counts.get("high", 0) > 0:
            suggestions.append("Address high-priority security and performance issues")
        if issue_counts.get("medium", 0) > 0:
            suggestions.append("Consider refactoring for better maintainability")

        # Specific suggestions
        if len(code.split('\n')) < 10:
            suggestions.append("Consider adding more comments for clarity")

        return suggestions

    def _calculate_complexity_score(self, issues: List[QualityIssue]) -> float:
        """Calculate complexity score (lower is better) using pre-defined weights."""
        if not issues:
            return 0.1

        # Use pre-defined severity weights for O(1) lookup
        score = sum(_SEVERITY_WEIGHTS.get(issue.level.value, 0.1) for issue in issues)
        return min(score / len(issues), 1.0)

    def _calculate_readability_score(self, code: str, issues: List[QualityIssue]) -> float:
        """Calculate readability score (higher is better)."""
        base_score = 0.8

        # Penalize for issues
        penalty = len(issues) * 0.05
        score = base_score - penalty

        # Bonus for comments - cache split result
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        total_lines = len(lines)
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            score += comment_ratio * 0.2

        return max(0.0, min(score, 1.0))

    def _calculate_maintainability_score(self, issues: List[QualityIssue]) -> float:
        """Calculate maintainability score (higher is better)."""
        base_score = 0.85

        # Efficient single-pass counting for complexity and naming issues
        complexity_count = 0
        naming_count = 0
        for issue in issues:
            msg_lower = issue.message.lower()
            if "complex" in msg_lower:
                complexity_count += 1
            if "name" in msg_lower or "case" in msg_lower:
                naming_count += 1

        penalty = (complexity_count + naming_count) * 0.1
        return max(0.0, min(base_score - penalty, 1.0))

    def clear_cache(self) -> None:
        """Clear the AST parsing cache."""
        self._parse_code_cached.cache_clear()
        self._ast_cache.clear()