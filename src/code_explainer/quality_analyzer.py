"""
Code quality analysis utilities.
"""

from typing import Dict, Any, List
import ast
import re
from enum import Enum


class IssueLevel(Enum):
    """Severity levels for code quality issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityIssue:
    """Represents a code quality issue."""

    def __init__(self, level, message: str, rule_id: str, line: int, column: int = 0):
        self.level = level
        self.message = message
        self.rule_id = rule_id
        self.line = line
        self.column = column


class CodeQualityAnalyzer:
    """Analyzes code quality and provides suggestions."""

    def __init__(self):
        pass

    def analyze_code(self, code: str) -> List[QualityIssue]:
        """Analyze code quality - main method expected by tests."""
        return self.analyze_quality(code)["issues"]

    def analyze_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality."""
        issues = []
        suggestions = []

        try:
            # Parse the code
            tree = ast.parse(code)

            # Check for various quality issues
            issues.extend(self._check_naming_conventions(tree))
            issues.extend(self._check_complexity(tree))
            issues.extend(self._check_best_practices(tree))

            # Generate suggestions
            suggestions = self._generate_suggestions(issues, code)

        except SyntaxError as e:
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                message=f"Syntax error: {e.msg}",
                rule_id="syntax_error",
                line=e.lineno or 1,
                column=e.offset or 0
            ))
            suggestions.append("Fix syntax errors before analysis")

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
        """Check naming conventions."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues.append(QualityIssue(
                        level=IssueLevel.MEDIUM,
                        message=f"Function '{node.name}' should use snake_case",
                        rule_id="naming_convention",
                        line=node.lineno,
                        column=node.col_offset
                    ))
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append(QualityIssue(
                        level=IssueLevel.MEDIUM,
                        message=f"Class '{node.name}' should use PascalCase",
                        rule_id="naming_convention",
                        line=node.lineno,
                        column=node.col_offset
                    ))

        return issues

    def _check_complexity(self, tree: ast.AST) -> List[QualityIssue]:
        """Check code complexity."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count statements roughly
                statements = len([n for n in ast.walk(node) if isinstance(n, (ast.Assign, ast.Return, ast.If, ast.For, ast.While))])
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
        """Calculate complexity score (lower is better)."""
        if not issues:
            return 0.1

        # Weight issues by severity
        score = 0
        for issue in issues:
            if issue.level == IssueLevel.CRITICAL:
                score += 1.0
            elif issue.level == IssueLevel.HIGH:
                score += 0.7
            elif issue.level == IssueLevel.MEDIUM:
                score += 0.4
            elif issue.level == IssueLevel.LOW:
                score += 0.1

        return min(score / len(issues), 1.0)

    def _calculate_readability_score(self, code: str, issues: List[QualityIssue]) -> float:
        """Calculate readability score (higher is better)."""
        base_score = 0.8

        # Penalize for issues
        penalty = len(issues) * 0.05
        score = base_score - penalty

        # Bonus for comments
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        total_lines = len(code.split('\n'))
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            score += comment_ratio * 0.2

        return max(0.0, min(score, 1.0))

    def _calculate_maintainability_score(self, issues: List[QualityIssue]) -> float:
        """Calculate maintainability score (higher is better)."""
        base_score = 0.85

        # Penalize for complexity and naming issues
        complexity_issues = [i for i in issues if "complex" in i.message.lower()]
        naming_issues = [i for i in issues if "name" in i.message.lower() or "case" in i.message.lower()]

        penalty = (len(complexity_issues) + len(naming_issues)) * 0.1
        score = base_score - penalty

        return max(0.0, min(score, 1.0))

from typing import Dict, Any, List
import ast
import re
from enum import Enum


class IssueLevel(Enum):
    """Severity levels for code quality issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CodeQualityAnalyzer:
    """Analyzes code quality and provides suggestions."""

    def __init__(self):
        pass

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code quality - main method expected by tests."""
        return self.analyze_quality(code)

    def analyze_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality."""
        issues = []
        suggestions = []

        try:
            # Parse the code
            tree = ast.parse(code)

            # Check for various quality issues
            issues.extend(self._check_naming_conventions(tree))
            issues.extend(self._check_complexity(tree))
            issues.extend(self._check_best_practices(tree))

            # Generate suggestions
            suggestions = self._generate_suggestions(issues, code)

        except SyntaxError:
            issues.append({
                "level": IssueLevel.CRITICAL,
                "message": "Syntax error in code",
                "line": 1
            })
            suggestions.append("Fix syntax errors before analysis")

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

    def _check_naming_conventions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check naming conventions."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues.append({
                        "level": IssueLevel.MEDIUM,
                        "message": f"Function '{node.name}' should use snake_case",
                        "line": node.lineno
                    })
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append({
                        "level": IssueLevel.MEDIUM,
                        "message": f"Class '{node.name}' should use PascalCase",
                        "line": node.lineno
                    })

        return issues

    def _check_complexity(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check code complexity."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count statements roughly
                statements = len([n for n in ast.walk(node) if isinstance(n, (ast.Assign, ast.Return, ast.If, ast.For, ast.While))])
                if statements > 20:
                    issues.append({
                        "level": IssueLevel.MEDIUM,
                        "message": f"Function '{node.name}' is too complex ({statements} statements)",
                        "line": node.lineno
                    })

        return issues

    def _check_best_practices(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check best practices."""
        issues = []

        # Check for eval usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'eval':
                issues.append({
                    "level": IssueLevel.HIGH,
                    "message": "Use of eval() is dangerous and should be avoided",
                    "line": node.lineno
                })

        return issues

    def _generate_suggestions(self, issues: List[Dict[str, Any]], code: str) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        if not issues:
            suggestions.append("Code follows good practices")
            return suggestions

        # Group issues by type
        issue_counts = {}
        for issue in issues:
            level = issue["level"].value if hasattr(issue["level"], 'value') else str(issue["level"])
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

    def _calculate_complexity_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate complexity score (lower is better)."""
        if not issues:
            return 0.1

        # Weight issues by severity
        score = 0
        for issue in issues:
            level = issue["level"].value if hasattr(issue["level"], 'value') else str(issue["level"])
            if level == "critical":
                score += 1.0
            elif level == "high":
                score += 0.7
            elif level == "medium":
                score += 0.4
            elif level == "low":
                score += 0.1

        return min(score / len(issues), 1.0)

    def _calculate_readability_score(self, code: str, issues: List[Dict[str, Any]]) -> float:
        """Calculate readability score (higher is better)."""
        base_score = 0.8

        # Penalize for issues
        penalty = len(issues) * 0.05
        score = base_score - penalty

        # Bonus for comments
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        total_lines = len(code.split('\n'))
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            score += comment_ratio * 0.2

        return max(0.0, min(score, 1.0))

    def _calculate_maintainability_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate maintainability score (higher is better)."""
        base_score = 0.85

        # Penalize for complexity and naming issues
        complexity_issues = [i for i in issues if "complex" in i["message"].lower()]
        naming_issues = [i for i in issues if "name" in i["message"].lower() or "case" in i["message"].lower()]

        penalty = (len(complexity_issues) + len(naming_issues)) * 0.1
        score = base_score - penalty

        return max(0.0, min(score, 1.0))