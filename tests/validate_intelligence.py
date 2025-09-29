"""Simple test runner for validating core intelligent features."""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")

    try:
        from src.code_explainer.device_manager import DeviceManager
        print("  ✅ DeviceManager imported")

        from src.code_explainer.intelligent_explainer import (
            IntelligentExplanationGenerator,
            ExplanationAudience,
            ExplanationStyle
        )
        print("  ✅ Intelligent explainer imported")

        from src.code_explainer.enhanced_language_processor import (
            EnhancedLanguageProcessor,
            CodeLanguage
        )
        print("  ✅ Enhanced language processor imported")

        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_device_manager():
    """Test device manager functionality."""
    print("🔧 Testing device manager...")

    try:
        from src.code_explainer.device_manager import DeviceManager

        manager = DeviceManager()
        current = manager.get_device()
        print(f"  ✅ Current device: {current}")

        # Test device compatibility
        is_compatible = manager.check_device_compatibility("cpu")
        print(f"  ✅ CPU compatibility: {is_compatible}")

        return True
    except Exception as e:
        print(f"  ❌ Device manager test failed: {e}")
        return False

def test_intelligent_explanation():
    """Test intelligent explanation generation."""
    print("🧠 Testing intelligent explanation...")

    try:
        from src.code_explainer.intelligent_explainer import (
            IntelligentExplanationGenerator,
            ExplanationAudience,
            ExplanationStyle
        )

        generator = IntelligentExplanationGenerator()

        test_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

        # Test basic explanation
        explanation = generator.explain_code(
            code=test_code,
            audience=ExplanationAudience.BEGINNER,
            style=ExplanationStyle.DETAILED
        )

        print(f"  ✅ Generated explanation ({len(explanation.primary_explanation)} chars)")
        print(f"  ✅ Language info: {explanation.language_info[:50]}...")

        # Test formatting
        formatted = generator.format_explanation(explanation, "markdown")
        print(f"  ✅ Formatted as markdown ({len(formatted)} chars)")

        return True
    except Exception as e:
        print(f"  ❌ Intelligent explanation test failed: {e}")
        return False

def test_language_processor():
    """Test enhanced language processor."""
    print("🔍 Testing language processor...")

    try:
        from src.code_explainer.enhanced_language_processor import (
            EnhancedLanguageProcessor,
            CodeLanguage
        )

        processor = EnhancedLanguageProcessor()

        test_code = """
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value
"""

        analysis = processor.analyze_code(test_code)

        print(f"  ✅ Language: {analysis.language.value}")
        print(f"  ✅ Confidence: {analysis.confidence:.2f}")
        print(f"  ✅ Functions: {len(analysis.functions)}")
        print(f"  ✅ Classes: {len(analysis.classes)}")

        return True
    except Exception as e:
        print(f"  ❌ Language processor test failed: {e}")
        return False

def test_performance():
    """Test performance of core operations."""
    print("⚡ Testing performance...")

    try:
        from src.code_explainer.intelligent_explainer import (
            IntelligentExplanationGenerator,
            ExplanationAudience,
            ExplanationStyle
        )

        generator = IntelligentExplanationGenerator()

        test_code = "def hello(): return 'Hello World'"

        # Time explanation generation
        start_time = time.time()
        explanation = generator.explain_code(
            code=test_code,
            audience=ExplanationAudience.BEGINNER,
            style=ExplanationStyle.CONCISE
        )
        duration = time.time() - start_time

        print(f"  ✅ Explanation generated in {duration:.3f}s")

        # Test multiple calls for consistency
        times = []
        for i in range(5):
            start = time.time()
            generator.explain_code(
                code=f"x = {i} + 1",
                audience=ExplanationAudience.EXPERT,
                style=ExplanationStyle.CONCISE
            )
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        print(f"  ✅ Average time over 5 calls: {avg_time:.3f}s")

        return True
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    print("🚀 Code Explainer Intelligence Features Validation")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Device Manager", test_device_manager),
        ("Intelligent Explanation", test_intelligent_explanation),
        ("Language Processor", test_language_processor),
        ("Performance", test_performance),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Intelligence features are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)