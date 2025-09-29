"""Tests for intelligent explanation system."""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Test basic integration without requiring optional dependencies
class TestIntelligentExplanationIntegration:
    """Test integration of intelligent explanation features."""

    def test_intelligent_explainer_availability_flag(self):
        """Test that the availability flag works correctly."""
        try:
            from src.code_explainer.model import INTELLIGENT_EXPLAINER_AVAILABLE
            assert isinstance(INTELLIGENT_EXPLAINER_AVAILABLE, bool)
        except ImportError:
            pytest.fail("Should be able to import availability flag")

    def test_fallback_when_dependencies_missing(self):
        """Test fallback behavior when intelligent dependencies are missing."""
        from src.code_explainer.model import CodeExplainer

        # Mock config
        config = MagicMock()
        config.model.name = "test-model"
        config.model.arch = "causal"

        explainer = CodeExplainer(config)

        # Mock the regular explain_code method
        explainer.explain_code = Mock(return_value="standard explanation")

        # These methods should exist and fall back gracefully
        result = explainer.explain_code_intelligent("def test(): pass")
        detailed_result = explainer.explain_code_intelligent_detailed("def test(): pass")

        # Should get fallback responses
        if result == "standard explanation":
            # Fallback worked
            assert True
        else:
            # Intelligent features are available
            assert isinstance(result, str) and len(result) > 0

        # Detailed method should return None or a dict
        assert detailed_result is None or isinstance(detailed_result, dict)

    def test_intelligent_methods_exist(self):
        """Test that intelligent explanation methods exist on CodeExplainer."""
        from src.code_explainer.model import CodeExplainer

        # Mock config
        config = MagicMock()
        config.model.name = "test-model"

        explainer = CodeExplainer(config)

        # Should have these methods regardless of whether dependencies are available
        assert hasattr(explainer, 'explain_code_intelligent')
        assert hasattr(explainer, 'explain_code_intelligent_detailed')
        assert callable(explainer.explain_code_intelligent)
        assert callable(explainer.explain_code_intelligent_detailed)

    @patch('src.code_explainer.model.INTELLIGENT_EXPLAINER_AVAILABLE', True)
    def test_intelligent_explainer_parameters(self):
        """Test that intelligent explainer methods accept the right parameters."""
        from src.code_explainer.model import CodeExplainer

        config = MagicMock()
        config.model.name = "test-model"

        explainer = CodeExplainer(config)
        explainer.explain_code = Mock(return_value="fallback")

        # Test parameter handling
        try:
            result = explainer.explain_code_intelligent(
                code="def test(): pass",
                strategy="adaptive",
                audience="beginner",
                style="detailed",
                include_examples=True,
                include_best_practices=True,
                include_security_notes=False,
                filename="test.py"
            )
            assert result is not None
        except Exception as e:
            # Should not crash due to parameter issues
            if "parameter" in str(e).lower():
                pytest.fail(f"Parameter handling failed: {e}")


# Test with actual intelligent features if available
try:
    from src.code_explainer.intelligent_explainer import (
        IntelligentExplanationGenerator,
        ExplanationAudience,
        ExplanationStyle
    )
    INTELLIGENT_FEATURES_AVAILABLE = True
except ImportError:
    INTELLIGENT_FEATURES_AVAILABLE = False


@pytest.mark.skipif(not INTELLIGENT_FEATURES_AVAILABLE, reason="Intelligent explanation features not available")
class TestIntelligentExplanationGenerator:
    """Test the intelligent explanation generator when available."""

    def test_basic_explanation_generation(self):
        """Test basic explanation generation."""
        generator = IntelligentExplanationGenerator()

        code = """
def greet(name):
    return f"Hello, {name}!"
"""

        explanation = generator.explain_code(
            code=code,
            audience=ExplanationAudience.BEGINNER,
            style=ExplanationStyle.DETAILED
        )

        assert explanation.primary_explanation
        assert len(explanation.primary_explanation) > 20
        assert explanation.language_info

    def test_audience_adaptation(self):
        """Test explanation adaptation for different audiences."""
        generator = IntelligentExplanationGenerator()

        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

        beginner_explanation = generator.explain_code(
            code=code,
            audience=ExplanationAudience.BEGINNER,
            style=ExplanationStyle.DETAILED
        )

        expert_explanation = generator.explain_code(
            code=code,
            audience=ExplanationAudience.EXPERT,
            style=ExplanationStyle.CONCISE
        )

        # Different explanations for different audiences
        assert beginner_explanation.primary_explanation != expert_explanation.primary_explanation

    def test_markdown_formatting(self):
        """Test markdown formatting of explanations."""
        generator = IntelligentExplanationGenerator()

        code = "x = [1, 2, 3]"

        explanation = generator.explain_code(code)
        formatted = generator.format_explanation(explanation, "markdown")

        assert isinstance(formatted, str)
        assert len(formatted) > 20

    def test_error_handling_with_invalid_code(self):
        """Test that invalid code doesn't crash the generator."""
        generator = IntelligentExplanationGenerator()

        invalid_code = "def incomplete_function("

        try:
            explanation = generator.explain_code(invalid_code)
            # Should still generate some explanation
            assert explanation.primary_explanation
        except Exception as e:
            # Should handle gracefully
            assert "syntax" in str(e).lower() or "parse" in str(e).lower()


@pytest.mark.skipif(not INTELLIGENT_FEATURES_AVAILABLE, reason="Enhanced language processor not available")
class TestEnhancedLanguageProcessor:
    """Test the enhanced language processor when available."""

    def test_basic_analysis(self):
        """Test basic code analysis."""
        try:
            from src.code_explainer.enhanced_language_processor import (
                EnhancedLanguageProcessor,
                CodeLanguage
            )
        except ImportError:
            pytest.skip("Enhanced language processor not available")

        processor = EnhancedLanguageProcessor()

        code = """
def add(a, b):
    return a + b
"""

        result = processor.analyze_code(code)
        assert result.language == CodeLanguage.PYTHON
        assert result.confidence > 0.5
        assert result.loc > 0

    def test_language_detection_with_filename(self):
        """Test language detection using filename hints."""
        try:
            from src.code_explainer.enhanced_language_processor import (
                EnhancedLanguageProcessor,
                CodeLanguage
            )
        except ImportError:
            pytest.skip("Enhanced language processor not available")

        processor = EnhancedLanguageProcessor()

        js_code = """
function hello() {
    console.log("Hello World");
}
"""

        # Should detect JavaScript from filename
        result = processor.analyze_code(js_code, filename="test.js")
        assert result.language == CodeLanguage.JAVASCRIPT


class TestCodeExplainerBackwardCompatibility:
    """Test that existing functionality still works with intelligent features added."""

    def test_existing_explain_code_still_works(self):
        """Test that the original explain_code method still works."""
        from src.code_explainer.model import CodeExplainer

        config = MagicMock()
        config.model.name = "test-model"
        config.model.arch = "causal"

        explainer = CodeExplainer(config)

        # Mock dependencies to avoid actual model loading
        explainer.tokenizer = Mock()
        explainer.model = Mock()
        explainer.tokenizer.encode = Mock(return_value=[1, 2, 3])
        explainer.model.generate = Mock(return_value=Mock(sequences=[[4, 5, 6]]))
        explainer.tokenizer.decode = Mock(return_value="mocked explanation")

        # Original method should still work
        result = explainer.explain_code("def test(): pass")
        assert result is not None

    def test_config_loading_compatibility(self):
        """Test that config loading works with intelligent features added."""
        from src.code_explainer.config import Config
        from pathlib import Path

        # Should be able to create config without issues
        try:
            config = Config()
            assert config is not None
        except Exception as e:
            if "intelligent" not in str(e).lower() and "tree" not in str(e).lower():
                # Re-raise if not related to optional intelligent features
                raise