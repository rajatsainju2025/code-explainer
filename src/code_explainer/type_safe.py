"""Type-safe wrapper for existing code explainer components."""

import logging
from typing import List, Optional, Dict, Any, Union, cast
from pathlib import Path

from .types import (
    ExplanationStrategy, ExplanationResult, ExplanationMetadata,
    SecurityValidationResult, CodeAnalysisResult, SecurityRiskLevel,
    BaseExplainer, ensure_strategy, ensure_risk_level,
    CodeExplainerError, ValidationError, SecurityError
)

logger = logging.getLogger(__name__)


class TypeSafeCodeExplainer(BaseExplainer):
    """Type-safe wrapper for the main CodeExplainer class."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize type-safe code explainer.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._base_explainer = self._initialize_base_explainer()

    def _initialize_base_explainer(self) -> Any:
        """Initialize the base explainer with error handling."""
        try:
            from code_explainer import CodeExplainer
            if self.config_path:
                return CodeExplainer(config_path=self.config_path)
            else:
                return CodeExplainer()
        except Exception as e:
            logger.error(f"Failed to initialize base explainer: {e}")
            raise CodeExplainerError(f"Initialization failed: {e}") from e

    def explain(
        self,
        code: str,
        strategy: Union[str, ExplanationStrategy] = ExplanationStrategy.ENHANCED_RAG
    ) -> ExplanationResult:
        """Explain code with type safety.

        Args:
            code: Python code to explain
            strategy: Explanation strategy to use

        Returns:
            Typed explanation result

        Raises:
            ValidationError: If input is invalid
            CodeExplainerError: If explanation fails
        """
        # Validate inputs
        if not code or not code.strip():
            raise ValidationError("Code cannot be empty")

        if len(code) > 100000:  # 100KB limit
            raise ValidationError("Code too large (max 100KB)")

        strategy_enum = ensure_strategy(strategy)

        import time
        start_time = time.time()

        try:
            # Try to use the explain method if available
            if hasattr(self._base_explainer, 'explain'):
                explanation = self._base_explainer.explain(code, strategy_enum.value)
            else:
                # Fallback implementation
                explanation = self._fallback_explanation(code, strategy_enum)

            execution_time = (time.time() - start_time) * 1000

            # Create metadata
            metadata = ExplanationMetadata(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                code_length=len(code),
                strategy=strategy_enum,
                processing_time_ms=execution_time
            )

            return ExplanationResult(
                explanation=explanation,
                strategy=strategy_enum,
                execution_time_ms=execution_time,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Explanation failed for strategy {strategy_enum}: {e}")
            raise CodeExplainerError(f"Explanation failed: {e}") from e

    def batch_explain(
        self,
        codes: List[str],
        strategy: Union[str, ExplanationStrategy] = ExplanationStrategy.ENHANCED_RAG
    ) -> List[ExplanationResult]:
        """Explain multiple code snippets.

        Args:
            codes: List of code snippets
            strategy: Explanation strategy

        Returns:
            List of explanation results
        """
        if not codes:
            raise ValidationError("Codes list cannot be empty")

        if len(codes) > 100:
            raise ValidationError("Too many codes (max 100)")

        strategy_enum = ensure_strategy(strategy)
        results = []

        for i, code in enumerate(codes):
            try:
                result = self.explain(code, strategy_enum)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to explain code {i}: {e}")
                # Create error result
                import time
                error_result = ExplanationResult(
                    explanation=f"Error: {str(e)}",
                    strategy=strategy_enum,
                    execution_time_ms=0,
                    metadata=ExplanationMetadata(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        code_length=len(code),
                        strategy=strategy_enum,
                        processing_time_ms=0
                    )
                )
                results.append(error_result)

        return results

    def validate_security(self, code: str) -> SecurityValidationResult:
        """Validate code security with type safety.

        Args:
            code: Code to validate

        Returns:
            Security validation result
        """
        if not code or not code.strip():
            raise ValidationError("Code cannot be empty")

        import time
        start_time = time.time()

        try:
            from code_explainer.security import CodeSecurityValidator

            validator = CodeSecurityValidator()

            if hasattr(validator, 'validate_code'):
                result = validator.validate_code(code)

                # Handle different result formats
                if isinstance(result, dict):
                    is_safe = result.get("is_safe", True)
                    issues = result.get("issues", [])
                    recommendations = result.get("recommendations", [])
                elif isinstance(result, tuple):
                    is_safe = result[0] if len(result) > 0 else True
                    issues = result[1] if len(result) > 1 else []
                    recommendations = result[2] if len(result) > 2 else []
                else:
                    is_safe = bool(result)
                    issues = []
                    recommendations = []
            else:
                # Fallback security check
                is_safe, issues, recommendations = self._fallback_security_check(code)

            # Determine risk level
            risk_level = SecurityRiskLevel.LOW
            if len(issues) >= 3:
                risk_level = SecurityRiskLevel.CRITICAL
            elif len(issues) >= 2:
                risk_level = SecurityRiskLevel.HIGH
            elif len(issues) >= 1:
                risk_level = SecurityRiskLevel.MEDIUM

            scan_time = (time.time() - start_time) * 1000

            return SecurityValidationResult(
                is_safe=is_safe,
                issues=issues,
                recommendations=recommendations,
                risk_level=risk_level,
                scan_time_ms=scan_time
            )

        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            scan_time = (time.time() - start_time) * 1000

            return SecurityValidationResult(
                is_safe=False,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Manual security review required"],
                risk_level=SecurityRiskLevel.UNKNOWN,
                scan_time_ms=scan_time
            )

    def analyze_code(self, code: str) -> CodeAnalysisResult:
        """Analyze code structure with type safety.

        Args:
            code: Code to analyze

        Returns:
            Code analysis result
        """
        if not code or not code.strip():
            raise ValidationError("Code cannot be empty")

        import ast
        from collections import defaultdict

        try:
            tree = ast.parse(code)

            # Count different node types
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1

            # Calculate quality metrics
            complexity_score = sum(node_counts.values())
            function_count = node_counts.get('FunctionDef', 0)
            class_count = node_counts.get('ClassDef', 0)
            line_count = len(code.splitlines())

            quality_metrics = {
                "cyclomatic_complexity": min(complexity_score // 10, 20),
                "maintainability_index": max(100 - line_count // 10, 0),
                "code_duplication": 0,  # Simplified
                "technical_debt_ratio": min(complexity_score / max(line_count, 1) * 100, 100)
            }

            # Generate suggestions
            suggestions = []
            if quality_metrics["cyclomatic_complexity"] > 10:
                suggestions.append("Consider breaking down complex functions")
            if line_count > 100:
                suggestions.append("Consider splitting into smaller modules")
            if function_count == 0 and class_count == 0:
                suggestions.append("Consider organizing code into functions or classes")

            return CodeAnalysisResult(
                complexity_score=complexity_score,
                function_count=function_count,
                class_count=class_count,
                line_count=line_count,
                has_imports=(node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0)) > 0,
                has_loops=(node_counts.get('For', 0) + node_counts.get('While', 0)) > 0,
                ast_valid=True,
                quality_metrics=quality_metrics,
                suggestions=suggestions
            )

        except SyntaxError as e:
            return CodeAnalysisResult(
                complexity_score=0,
                function_count=0,
                class_count=0,
                line_count=len(code.splitlines()),
                has_imports=False,
                has_loops=False,
                ast_valid=False,
                quality_metrics={"syntax_error": str(e)},
                suggestions=["Fix syntax errors before analysis"]
            )

    def get_supported_strategies(self) -> List[ExplanationStrategy]:
        """Get list of supported explanation strategies.

        Returns:
            List of supported strategies
        """
        return list(ExplanationStrategy)

    def _fallback_explanation(self, code: str, strategy: ExplanationStrategy) -> str:
        """Fallback explanation when main explainer is unavailable.

        Args:
            code: Code to explain
            strategy: Strategy to use

        Returns:
            Fallback explanation
        """
        analysis = self.analyze_code(code)

        explanation_parts = [
            f"Code Analysis (using {strategy.value} strategy):",
            f"- Lines of code: {analysis['line_count']}",
            f"- Functions: {analysis['function_count']}",
            f"- Classes: {analysis['class_count']}",
            f"- Complexity score: {analysis['complexity_score']}"
        ]

        if analysis['has_imports']:
            explanation_parts.append("- Contains imports")

        if analysis['has_loops']:
            explanation_parts.append("- Contains loops")

        if analysis['suggestions']:
            explanation_parts.append("\nSuggestions:")
            for suggestion in analysis['suggestions']:
                explanation_parts.append(f"- {suggestion}")

        # Add code snippet analysis
        lines = code.strip().splitlines()
        if lines:
            explanation_parts.append(f"\nCode snippet preview: {lines[0][:50]}...")

        return "\n".join(explanation_parts)

    def _fallback_security_check(self, code: str) -> tuple[bool, List[str], List[str]]:
        """Fallback security check when validator is unavailable.

        Args:
            code: Code to check

        Returns:
            Tuple of (is_safe, issues, recommendations)
        """
        issues = []
        recommendations = []

        # Basic pattern matching
        dangerous_patterns = [
            ("eval(", "Dangerous eval() usage"),
            ("exec(", "Dangerous exec() usage"),
            ("__import__", "Dynamic import usage"),
            ("open(", "File access detected"),
            ("subprocess", "Subprocess usage"),
            ("os.system", "System command execution"),
        ]

        code_lower = code.lower()
        for pattern, issue in dangerous_patterns:
            if pattern in code_lower:
                issues.append(issue)
                recommendations.append(f"Review usage of {pattern}")

        # Check for long code that might be obfuscated
        if len(code) > 10000:
            issues.append("Very long code detected")
            recommendations.append("Review for potential obfuscation")

        is_safe = len(issues) == 0

        if not is_safe:
            recommendations.append("Consider using a more secure alternative")
            recommendations.append("Review code manually for security issues")

        return is_safe, issues, recommendations


# Type-safe factory function
def create_type_safe_explainer(config_path: Optional[str] = None) -> TypeSafeCodeExplainer:
    """Create a type-safe code explainer instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Type-safe code explainer instance
    """
    return TypeSafeCodeExplainer(config_path)


# Convenience functions with type safety
def explain_code_safely(
    code: str,
    strategy: Union[str, ExplanationStrategy] = ExplanationStrategy.ENHANCED_RAG,
    config_path: Optional[str] = None
) -> ExplanationResult:
    """Safely explain code with full type checking.

    Args:
        code: Code to explain
        strategy: Explanation strategy
        config_path: Optional config path

    Returns:
        Explanation result
    """
    explainer = create_type_safe_explainer(config_path)
    return explainer.explain(code, strategy)


def validate_code_safely(
    code: str,
    config_path: Optional[str] = None
) -> SecurityValidationResult:
    """Safely validate code security.

    Args:
        code: Code to validate
        config_path: Optional config path

    Returns:
        Security validation result
    """
    explainer = create_type_safe_explainer(config_path)
    return explainer.validate_security(code)


def analyze_code_safely(
    code: str,
    config_path: Optional[str] = None
) -> CodeAnalysisResult:
    """Safely analyze code structure.

    Args:
        code: Code to analyze
        config_path: Optional config path

    Returns:
        Code analysis result
    """
    explainer = create_type_safe_explainer(config_path)
    return explainer.analyze_code(code)
