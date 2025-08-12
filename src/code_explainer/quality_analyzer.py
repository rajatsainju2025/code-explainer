"""Code quality analysis utilities."""

import ast
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IssueLevel(Enum):
    """Severity levels for code issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CodeIssue:
    """Represents a code quality issue."""

    level: IssueLevel
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    rule_id: Optional[str] = None
    suggestion: Optional[str] = None


class CodeQualityAnalyzer:
    """Analyzes Python code for quality issues and best practices."""

    def __init__(self):
        """Initialize the code quality analyzer."""
        self.issues: List[CodeIssue] = []

    def analyze_code(self, code: str) -> List[CodeIssue]:
        """Analyze code and return a list of issues.

        Args:
            code: Python code to analyze

        Returns:
            List of CodeIssue objects
        """
        self.issues.clear()

        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Run various checks
            self._check_syntax_style(code)
            self._check_ast_patterns(tree)
            self._check_naming_conventions(tree)
            self._check_complexity(tree)
            self._check_best_practices(tree, code)

        except SyntaxError as e:
            self.issues.append(
                CodeIssue(
                    level=IssueLevel.ERROR,
                    message=f"Syntax error: {e.msg}",
                    line_number=e.lineno,
                    column=e.offset,
                    rule_id="syntax_error",
                )
            )
        except Exception as e:
            logger.warning(f"Analysis failed: {e}")
            self.issues.append(
                CodeIssue(
                    level=IssueLevel.WARNING,
                    message=f"Analysis failed: {str(e)}",
                    rule_id="analysis_error",
                )
            )

        return self.issues

    def _check_syntax_style(self, code: str) -> None:
        """Check syntax and style issues."""
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:  # PEP 8 recommends 79, but 88 is more modern
                self.issues.append(
                    CodeIssue(
                        level=IssueLevel.WARNING,
                        message=f"Line too long ({len(line)} > 88 characters)",
                        line_number=i,
                        rule_id="line_too_long",
                        suggestion="Break the line into multiple lines",
                    )
                )

            # Check trailing whitespace
            if line.endswith(" ") or line.endswith("\t"):
                self.issues.append(
                    CodeIssue(
                        level=IssueLevel.INFO,
                        message="Trailing whitespace",
                        line_number=i,
                        rule_id="trailing_whitespace",
                        suggestion="Remove trailing whitespace",
                    )
                )

            # Check for tab characters
            if "\t" in line:
                self.issues.append(
                    CodeIssue(
                        level=IssueLevel.WARNING,
                        message="Tab character found, use spaces for indentation",
                        line_number=i,
                        rule_id="tab_indentation",
                        suggestion="Replace tabs with 4 spaces",
                    )
                )

    def _check_ast_patterns(self, tree: ast.AST) -> None:
        """Check AST patterns for potential issues."""
        for node in ast.walk(tree):
            # Check for bare except clauses
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self.issues.append(
                    CodeIssue(
                        level=IssueLevel.WARNING,
                        message="Bare except clause catches all exceptions",
                        line_number=node.lineno,
                        rule_id="bare_except",
                        suggestion="Specify the exception type or use 'except Exception:'",
                    )
                )

            # Check for unused variables (simple case)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("_"):
                        # Skip variables starting with underscore (convention for unused)
                        continue

            # Check for potential security issues
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == "eval":
                        self.issues.append(
                            CodeIssue(
                                level=IssueLevel.CRITICAL,
                                message="Use of 'eval()' is dangerous",
                                line_number=node.lineno,
                                rule_id="dangerous_eval",
                                suggestion="Consider safer alternatives like ast.literal_eval()",
                            )
                        )
                    elif node.func.id == "exec":
                        self.issues.append(
                            CodeIssue(
                                level=IssueLevel.CRITICAL,
                                message="Use of 'exec()' is dangerous",
                                line_number=node.lineno,
                                rule_id="dangerous_exec",
                                suggestion="Avoid dynamic code execution",
                            )
                        )

            # Check for missing docstrings
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    node_type = (
                        "function"
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else "class"
                    )
                    self.issues.append(
                        CodeIssue(
                            level=IssueLevel.INFO,
                            message=f"Missing docstring for {node_type} '{node.name}'",
                            line_number=node.lineno,
                            rule_id="missing_docstring",
                            suggestion=f"Add a docstring describing the {node_type}",
                        )
                    )

    def _check_naming_conventions(self, tree: ast.AST) -> None:
        """Check naming conventions (PEP 8)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Function names should be lowercase with underscores
                if not re.match(r"^[a-z_][a-z0-9_]*$", node.name):
                    self.issues.append(
                        CodeIssue(
                            level=IssueLevel.WARNING,
                            message=f"Function name '{node.name}' should be lowercase with underscores",
                            line_number=node.lineno,
                            rule_id="function_naming",
                            suggestion="Use snake_case for function names",
                        )
                    )

            elif isinstance(node, ast.ClassDef):
                # Class names should be CamelCase
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    self.issues.append(
                        CodeIssue(
                            level=IssueLevel.WARNING,
                            message=f"Class name '{node.name}' should be CamelCase",
                            line_number=node.lineno,
                            rule_id="class_naming",
                            suggestion="Use CamelCase for class names",
                        )
                    )

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Variable names should be lowercase with underscores
                if node.id.isupper() and len(node.id) > 1:
                    # Likely a constant, which is okay
                    continue
                elif not re.match(r"^[a-z_][a-z0-9_]*$", node.id) and not node.id.startswith("_"):
                    self.issues.append(
                        CodeIssue(
                            level=IssueLevel.INFO,
                            message=f"Variable name '{node.id}' should be lowercase with underscores",
                            line_number=node.lineno,
                            rule_id="variable_naming",
                            suggestion="Use snake_case for variable names",
                        )
                    )

    def _check_complexity(self, tree: ast.AST) -> None:
        """Check cyclomatic complexity and other complexity metrics."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > 10:
                    self.issues.append(
                        CodeIssue(
                            level=IssueLevel.WARNING,
                            message=f"Function '{node.name}' has high complexity ({complexity})",
                            line_number=node.lineno,
                            rule_id="high_complexity",
                            suggestion="Consider breaking this function into smaller functions",
                        )
                    )

                # Check for too many arguments
                args_count = len(node.args.args)
                if args_count > 5:
                    self.issues.append(
                        CodeIssue(
                            level=IssueLevel.WARNING,
                            message=f"Function '{node.name}' has too many arguments ({args_count})",
                            line_number=node.lineno,
                            rule_id="too_many_arguments",
                            suggestion="Consider using a dictionary or class for parameters",
                        )
                    )

    def _calculate_cyclomatic_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity

        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _check_best_practices(self, tree: ast.AST, code: str) -> None:
        """Check for Python best practices."""
        for node in ast.walk(tree):
            # Check for mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        self.issues.append(
                            CodeIssue(
                                level=IssueLevel.WARNING,
                                message=f"Mutable default argument in function '{node.name}'",
                                line_number=node.lineno,
                                rule_id="mutable_default",
                                suggestion="Use None as default and create the mutable object inside the function",
                            )
                        )

            # Check for string formatting
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                if isinstance(node.left, ast.Str):
                    self.issues.append(
                        CodeIssue(
                            level=IssueLevel.INFO,
                            message="Consider using f-strings or .format() instead of % formatting",
                            line_number=node.lineno,
                            rule_id="old_string_formatting",
                            suggestion="Use f-strings for better readability",
                        )
                    )

        # Check for TODO/FIXME comments
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if "TODO" in line or "FIXME" in line:
                self.issues.append(
                    CodeIssue(
                        level=IssueLevel.INFO,
                        message="TODO/FIXME comment found",
                        line_number=i,
                        rule_id="todo_comment",
                        suggestion="Consider creating an issue or implementing the TODO",
                    )
                )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        level_counts = {level.value: 0 for level in IssueLevel}
        rule_counts = {}

        for issue in self.issues:
            level_counts[issue.level.value] += 1
            rule_counts[issue.rule_id] = rule_counts.get(issue.rule_id, 0) + 1

        return {
            "total_issues": len(self.issues),
            "by_level": level_counts,
            "by_rule": rule_counts,
            "issues": [
                {
                    "level": issue.level.value,
                    "message": issue.message,
                    "line_number": issue.line_number,
                    "column": issue.column,
                    "rule_id": issue.rule_id,
                    "suggestion": issue.suggestion,
                }
                for issue in self.issues
            ],
        }


def analyze_code_quality(code: str) -> Dict[str, Any]:
    """Convenience function to analyze code quality.

    Args:
        code: Python code to analyze

    Returns:
        Dictionary with analysis results
    """
    analyzer = CodeQualityAnalyzer()
    analyzer.analyze_code(code)
    return analyzer.get_summary()


def main():
    """CLI entry point for code quality analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze Python code quality")
    parser.add_argument("file", help="Python file to analyze")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    try:
        with open(args.file, "r") as f:
            code = f.read()

        results = analyze_code_quality(code)

        if args.format == "json":
            output = json.dumps(results, indent=2)
        else:
            # Text format
            output_lines = [f"Code Quality Analysis for {args.file}"]
            output_lines.append("=" * 50)
            output_lines.append(f"Total issues: {results['total_issues']}")
            output_lines.append("")

            if results["total_issues"] > 0:
                output_lines.append("Issues by level:")
                for level, count in results["by_level"].items():
                    if count > 0:
                        output_lines.append(f"  {level.upper()}: {count}")

                output_lines.append("")
                output_lines.append("Detailed issues:")
                for issue in results["issues"]:
                    line_info = f" (line {issue['line_number']})" if issue["line_number"] else ""
                    output_lines.append(
                        f"  [{issue['level'].upper()}]{line_info} {issue['message']}"
                    )
                    if issue["suggestion"]:
                        output_lines.append(f"    Suggestion: {issue['suggestion']}")
            else:
                output_lines.append("No issues found! 🎉")

            output = "\n".join(output_lines)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Results saved to {args.output}")
        else:
            print(output)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
