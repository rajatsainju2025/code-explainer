"""Comprehensive evaluation system for intelligent code explanation features.

This module provides automated benchmarks, quality metrics, and
cross-device testing for the enhanced intelligent code explanation system.
"""

import os
import sys
import time
import json
import asyncio
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from tqdm import tqdm

# Import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from src.code_explainer.model import CodeExplainer, INTELLIGENT_EXPLAINER_AVAILABLE
    from src.code_explainer.config import Config, init_config
    from src.code_explainer.device_manager import DeviceManager

    # Import intelligent features if available
    if INTELLIGENT_EXPLAINER_AVAILABLE:
        from src.code_explainer.intelligent_explainer import (
            IntelligentExplanationGenerator,
            ExplanationAudience,
            ExplanationStyle
        )
        from src.code_explainer.enhanced_language_processor import (
            EnhancedLanguageProcessor,
            CodeLanguage
        )

except ImportError as e:
    print(f"Warning: Could not import code explainer modules: {e}")
    print("Trying alternative import paths...")

    try:
        # Alternative import without src prefix
        sys.path.insert(0, os.path.dirname(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

        from code_explainer.model import CodeExplainer, INTELLIGENT_EXPLAINER_AVAILABLE
        from code_explainer.config import Config, init_config
        from code_explainer.device_manager import DeviceManager

        # Import intelligent features if available
        INTELLIGENT_EXPLAINER_AVAILABLE = False  # Default to False for safety
        try:
            from code_explainer.intelligent_explainer import (
                IntelligentExplanationGenerator,
                ExplanationAudience,
                ExplanationStyle
            )
            from code_explainer.enhanced_language_processor import (
                EnhancedLanguageProcessor,
                CodeLanguage
            )
            INTELLIGENT_EXPLAINER_AVAILABLE = True
        except ImportError:
            pass

    except ImportError as e2:
        print(f"Failed alternative imports: {e2}")
        sys.exit(1)


@dataclass
class EvaluationResult:
    """Results from a single evaluation."""

    test_name: str
    method: str
    duration: float
    success: bool
    output_length: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    test_suite: str
    results: List[EvaluationResult]
    total_duration: float
    success_rate: float
    avg_response_time: float
    device_info: Dict[str, Any]


class CodeExplainerEvaluator:
    """Comprehensive evaluator for code explainer functionality."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize evaluator with configuration."""
        self.config_path = config_path or "configs/default.yaml"
        self.device_manager = DeviceManager()
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases for evaluation."""
        return [
            {
                "name": "simple_function",
                "code": """
def add_numbers(a, b):
    \"\"\"Add two numbers together.\"\"\"
    return a + b
""",
                "expected_keywords": ["function", "add", "parameters", "return"],
                "complexity": "simple",
                "language": "python"
            },
            {
                "name": "class_definition",
                "code": """
class Calculator:
    def __init__(self, precision=2):
        self.precision = precision

    def multiply(self, x, y):
        result = x * y
        return round(result, self.precision)
""",
                "expected_keywords": ["class", "constructor", "__init__", "method", "instance"],
                "complexity": "medium",
                "language": "python"
            },
            {
                "name": "complex_algorithm",
                "code": """
def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot = partition(arr, low, high)
        quicksort(arr, low, pivot - 1)
        quicksort(arr, pivot + 1, high)

    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
""",
                "expected_keywords": ["algorithm", "sorting", "recursive", "pivot", "partition"],
                "complexity": "complex",
                "language": "python"
            },
            {
                "name": "javascript_function",
                "code": """
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }

    let a = 0, b = 1, temp;
    for (let i = 2; i <= n; i++) {
        temp = a + b;
        a = b;
        b = temp;
    }

    return b;
}
""",
                "expected_keywords": ["function", "fibonacci", "iterative", "loop"],
                "complexity": "medium",
                "language": "javascript"
            },
            {
                "name": "framework_code",
                "code": """
from flask import Flask, jsonify, request
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)

@app.route('/users', methods=['GET'])
def get_users():
    return jsonify([])

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    return jsonify({"id": 1, "username": data.get("username")})
""",
                "expected_keywords": ["flask", "api", "database", "sqlalchemy", "route"],
                "complexity": "complex",
                "language": "python"
            }
        ]

    def evaluate_method(
        self,
        explainer: CodeExplainer,
        method_name: str,
        test_case: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """Evaluate a specific explanation method."""

        start_time = time.time()
        success = False
        output_length = 0
        error_message = None
        metadata = {}

        try:
            # Get the method from the explainer
            method = getattr(explainer, method_name)

            # Call the method with the test code
            if method_name in ["explain_code_intelligent", "explain_code_intelligent_detailed"]:
                result = method(test_case["code"], **kwargs)
            else:
                result = method(test_case["code"])

            if result is not None:
                success = True
                if isinstance(result, str):
                    output_length = len(result)
                    metadata["output_type"] = "string"
                elif isinstance(result, dict):
                    output_length = len(str(result))
                    metadata["output_type"] = "dict"
                    metadata["dict_keys"] = list(result.keys())
                else:
                    output_length = len(str(result))
                    metadata["output_type"] = type(result).__name__

                # Check for expected keywords
                output_str = str(result).lower()
                found_keywords = [kw for kw in test_case.get("expected_keywords", [])
                                if kw.lower() in output_str]
                metadata["found_keywords"] = found_keywords
                metadata["keyword_coverage"] = len(found_keywords) / max(1, len(test_case.get("expected_keywords", [])))

        except Exception as e:
            error_message = str(e)
            metadata["exception_type"] = type(e).__name__

        duration = time.time() - start_time

        return EvaluationResult(
            test_name=test_case["name"],
            method=method_name,
            duration=duration,
            success=success,
            output_length=output_length,
            error_message=error_message,
            metadata=metadata
        )

    def run_device_compatibility_test(self) -> Dict[str, Any]:
        """Test device compatibility across available devices."""

        device_results = {}
        available_devices = self.device_manager.get_available_devices()

        for device_name, device_info in available_devices.items():
            print(f"\n🔧 Testing device: {device_name}")

            try:
                # Test device selection
                self.device_manager.set_device(device_name)
                current_device = self.device_manager.get_current_device()

                # Test basic device operations
                test_tensor = self.device_manager.move_to_device([1, 2, 3, 4, 5])

                device_results[device_name] = {
                    "available": True,
                    "device_info": device_info,
                    "current_device": str(current_device),
                    "tensor_test": "passed",
                    "memory_info": self.device_manager.get_device_memory_info()
                }

            except Exception as e:
                device_results[device_name] = {
                    "available": False,
                    "error": str(e),
                    "device_info": device_info
                }

        return device_results

    def benchmark_explanation_methods(self, max_workers: int = 2) -> BenchmarkSuite:
        """Benchmark all available explanation methods."""

        print("🚀 Starting comprehensive benchmark suite...")

        # Get device info
        device_info = {
            "current_device": str(self.device_manager.get_current_device()),
            "available_devices": list(self.device_manager.get_available_devices().keys()),
            "memory_info": self.device_manager.get_device_memory_info()
        }

        # Define methods to test
        methods_to_test = [
            ("explain_code", {}),
            ("explain_code_intelligent", {"audience": "beginner", "style": "detailed"}),
            ("explain_code_intelligent", {"audience": "expert", "style": "concise"}),
            ("explain_code_intelligent_detailed", {}),
        ]

        # Only test intelligent methods if available
        if not INTELLIGENT_EXPLAINER_AVAILABLE:
            methods_to_test = [("explain_code", {})]

        all_results = []
        start_time = time.time()

        try:
            # Initialize explainer (this might take time for model loading)
            print("📦 Initializing CodeExplainer...")
            config = init_config(self.config_path)
            explainer = CodeExplainer(config)

            # Run evaluations
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for test_case in self.test_cases:
                    for method_name, kwargs in methods_to_test:
                        if hasattr(explainer, method_name):
                            future = executor.submit(
                                self.evaluate_method,
                                explainer, method_name, test_case, **kwargs
                            )
                            futures.append(future)

                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Running tests"):
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per test
                        all_results.append(result)
                    except Exception as e:
                        print(f"❌ Test failed with exception: {e}")

        except Exception as e:
            print(f"❌ Failed to initialize explainer: {e}")
            # Create a minimal failure result
            all_results.append(EvaluationResult(
                test_name="initialization",
                method="setup",
                duration=0.0,
                success=False,
                output_length=0,
                error_message=str(e)
            ))

        total_duration = time.time() - start_time

        # Calculate metrics
        successful_results = [r for r in all_results if r.success]
        success_rate = len(successful_results) / max(1, len(all_results))
        avg_response_time = statistics.mean([r.duration for r in all_results]) if all_results else 0.0

        return BenchmarkSuite(
            test_suite="comprehensive",
            results=all_results,
            total_duration=total_duration,
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            device_info=device_info
        )

    def evaluate_intelligent_features(self) -> Dict[str, Any]:
        """Evaluate intelligent explanation features specifically."""

        if not INTELLIGENT_EXPLAINER_AVAILABLE:
            return {
                "available": False,
                "message": "Intelligent explanation features not available"
            }

        print("🧠 Evaluating intelligent features...")

        try:
            # Test language processor
            processor = EnhancedLanguageProcessor()

            processor_results = {}
            for test_case in self.test_cases[:3]:  # Test first 3 cases
                try:
                    analysis = processor.analyze_code(test_case["code"])
                    processor_results[test_case["name"]] = {
                        "language": analysis.language.value,
                        "confidence": analysis.confidence,
                        "loc": analysis.loc,
                        "functions_count": len(analysis.functions),
                        "classes_count": len(analysis.classes),
                    }
                except Exception as e:
                    processor_results[test_case["name"]] = {"error": str(e)}

            # Test explanation generator
            generator = IntelligentExplanationGenerator()

            generator_results = {}
            test_audiences = [ExplanationAudience.BEGINNER, ExplanationAudience.EXPERT]
            test_styles = [ExplanationStyle.CONCISE, ExplanationStyle.DETAILED]

            for i, test_case in enumerate(self.test_cases[:2]):  # Test first 2 cases
                try:
                    audience = test_audiences[i % len(test_audiences)]
                    style = test_styles[i % len(test_styles)]

                    explanation = generator.explain_code(
                        test_case["code"],
                        audience=audience,
                        style=style
                    )

                    generator_results[test_case["name"]] = {
                        "primary_explanation_length": len(explanation.primary_explanation),
                        "has_language_info": bool(explanation.language_info),
                        "has_best_practices": bool(explanation.best_practices),
                        "has_examples": bool(explanation.examples),
                        "audience": audience.value,
                        "style": style.value
                    }
                except Exception as e:
                    generator_results[test_case["name"]] = {"error": str(e)}

            return {
                "available": True,
                "language_processor": processor_results,
                "explanation_generator": generator_results
            }

        except Exception as e:
            return {
                "available": True,
                "error": str(e)
            }

    def generate_report(
        self,
        benchmark_results: BenchmarkSuite,
        device_results: Dict[str, Any],
        intelligent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": len(benchmark_results.results),
                "success_rate": benchmark_results.success_rate,
                "avg_response_time": benchmark_results.avg_response_time,
                "total_duration": benchmark_results.total_duration,
                "intelligent_features_available": INTELLIGENT_EXPLAINER_AVAILABLE
            },
            "device_compatibility": device_results,
            "intelligent_features": intelligent_results,
            "benchmark_details": {
                "results": [asdict(r) for r in benchmark_results.results],
                "device_info": benchmark_results.device_info
            },
            "performance_metrics": self._calculate_performance_metrics(benchmark_results.results),
            "quality_metrics": self._calculate_quality_metrics(benchmark_results.results)
        }

        return report

    def _calculate_performance_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate performance metrics from results."""

        if not results:
            return {}

        durations = [r.duration for r in results if r.success]
        output_lengths = [r.output_length for r in results if r.success]

        return {
            "min_duration": min(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "median_duration": statistics.median(durations) if durations else 0.0,
            "avg_output_length": statistics.mean(output_lengths) if output_lengths else 0.0,
            "throughput_tests_per_second": len(durations) / sum(durations) if durations and sum(durations) > 0 else 0.0
        }

    def _calculate_quality_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate quality metrics from results."""

        successful_results = [r for r in results if r.success and r.metadata]

        if not successful_results:
            return {}

        keyword_coverages = [
            r.metadata.get("keyword_coverage", 0.0)
            for r in successful_results
            if r.metadata and "keyword_coverage" in r.metadata
        ]

        return {
            "avg_keyword_coverage": statistics.mean(keyword_coverages) if keyword_coverages else 0.0,
            "quality_score": statistics.mean(keyword_coverages) if keyword_coverages else 0.0,
            "output_consistency": len(set(r.metadata.get("output_type", "") for r in successful_results)) == 1
        }


def main():
    """Run comprehensive evaluation suite."""

    print("🔍 Code Explainer Comprehensive Evaluation Suite")
    print("=" * 60)

    # Initialize evaluator
    evaluator = CodeExplainerEvaluator()

    # Run device compatibility tests
    print("\n1. Device Compatibility Testing")
    device_results = evaluator.run_device_compatibility_test()

    # Run main benchmark
    print("\n2. Performance Benchmarking")
    benchmark_results = evaluator.benchmark_explanation_methods()

    # Run intelligent features evaluation
    print("\n3. Intelligent Features Evaluation")
    intelligent_results = evaluator.evaluate_intelligent_features()

    # Generate report
    print("\n4. Generating Report")
    report = evaluator.generate_report(benchmark_results, device_results, intelligent_results)

    # Save report
    report_path = Path("results") / f"evaluation_report_{int(time.time())}.json"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {report['summary']['total_tests'] * report['summary']['success_rate']:.0f}/{report['summary']['total_tests']}")
    print(f"⚡ Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"⏱️  Average Response Time: {report['summary']['avg_response_time']:.2f}s")
    print(f"🧠 Intelligent Features: {'✅ Available' if report['summary']['intelligent_features_available'] else '❌ Not Available'}")
    print(f"💾 Report saved to: {report_path}")

    # Print device compatibility
    print(f"\n🔧 Device Compatibility:")
    for device, info in device_results.items():
        status = "✅" if info.get("available", False) else "❌"
        print(f"  {status} {device}: {info.get('current_device', 'N/A')}")

    return report


if __name__ == "__main__":
    main()