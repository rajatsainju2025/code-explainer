"""Golden test datasets for model stability and regression testing."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)


@dataclass
class GoldenTestCase:
    """A single golden test case with input and expected output."""
    id: str
    category: str
    input_code: str
    expected_explanation: str
    strategy: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldenTestCase':
        """Create from dictionary."""
        return cls(**data)

    def get_input_hash(self) -> str:
        """Get hash of input for change detection."""
        content = f"{self.input_code}_{self.strategy or 'default'}"
        return hashlib.md5(content.encode()).hexdigest()


class GoldenTestDatasets:
    """Collection of golden test datasets organized by category."""

    def __init__(self, data_dir: str = "./golden_tests"):
        """Initialize golden test datasets.

        Args:
            data_dir: Directory to store golden test data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, List[GoldenTestCase]] = {}
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Load existing datasets from disk."""
        for category_file in self.data_dir.glob("*.json"):
            category = category_file.stem
            try:
                with open(category_file, 'r') as f:
                    data = json.load(f)
                    test_cases = [GoldenTestCase.from_dict(case) for case in data]
                    self.datasets[category] = test_cases
                    logger.info(f"Loaded {len(test_cases)} golden tests for category: {category}")
            except Exception as e:
                logger.error(f"Failed to load golden tests for {category}: {e}")

    def save_datasets(self) -> None:
        """Save all datasets to disk."""
        for category, test_cases in self.datasets.items():
            file_path = self.data_dir / f"{category}.json"
            try:
                data = [case.to_dict() for case in test_cases]
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved {len(test_cases)} golden tests for category: {category}")
            except Exception as e:
                logger.error(f"Failed to save golden tests for {category}: {e}")

    def add_test_case(self, test_case: GoldenTestCase) -> None:
        """Add a new test case to the appropriate category."""
        category = test_case.category
        if category not in self.datasets:
            self.datasets[category] = []

        # Check for duplicates
        existing_ids = {case.id for case in self.datasets[category]}
        if test_case.id in existing_ids:
            logger.warning(f"Test case {test_case.id} already exists in category {category}")
            return

        self.datasets[category].append(test_case)
        logger.info(f"Added golden test case {test_case.id} to category {category}")

    def get_category_tests(self, category: str) -> List[GoldenTestCase]:
        """Get all test cases for a specific category."""
        return self.datasets.get(category, [])

    def get_all_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self.datasets.keys())

    def get_total_test_count(self) -> int:
        """Get total number of test cases across all categories."""
        return sum(len(tests) for tests in self.datasets.values())


def create_core_golden_tests() -> GoldenTestDatasets:
    """Create core golden test datasets for fundamental functionality."""

    datasets = GoldenTestDatasets()

    # Basic function explanations
    basic_tests = [
        GoldenTestCase(
            id="basic_function_add",
            category="basic_functions",
            input_code="def add(a, b):\n    return a + b",
            expected_explanation="This function takes two parameters 'a' and 'b' and returns their sum. It performs simple addition operation.",
            strategy="vanilla"
        ),
        GoldenTestCase(
            id="basic_function_factorial",
            category="basic_functions",
            input_code="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            expected_explanation="This function calculates the factorial of a number 'n' using recursion. If n is less than or equal to 1, it returns 1 (base case). Otherwise, it returns n multiplied by the factorial of (n-1).",
            strategy="vanilla"
        ),
        GoldenTestCase(
            id="basic_loop_sum",
            category="basic_functions",
            input_code="def sum_list(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total",
            expected_explanation="This function calculates the sum of all numbers in a list. It initializes a total variable to 0, then iterates through each number in the input list, adding each number to the total, and finally returns the sum.",
            strategy="vanilla"
        )
    ]

    for test in basic_tests:
        datasets.add_test_case(test)

    # Data structure operations
    data_structure_tests = [
        GoldenTestCase(
            id="list_comprehension_squares",
            category="data_structures",
            input_code="squares = [x**2 for x in range(10)]",
            expected_explanation="This creates a list called 'squares' containing the square of each number from 0 to 9 using list comprehension. The result is [0, 1, 4, 9, 16, 25, 36, 49, 64, 81].",
            strategy="vanilla"
        ),
        GoldenTestCase(
            id="dict_comprehension_mapping",
            category="data_structures",
            input_code="word_lengths = {word: len(word) for word in ['hello', 'world', 'python']}",
            expected_explanation="This creates a dictionary called 'word_lengths' using dictionary comprehension. Each key is a word from the list, and each value is the length of that word. The result maps each word to its character count.",
            strategy="vanilla"
        ),
        GoldenTestCase(
            id="stack_operations",
            category="data_structures",
            input_code="stack = []\nstack.append(1)\nstack.append(2)\nitem = stack.pop()",
            expected_explanation="This code demonstrates basic stack operations using a Python list. It creates an empty list as a stack, pushes two items (1 and 2) onto the stack using append(), then pops the last item (2) from the stack and stores it in the variable 'item'.",
            strategy="vanilla"
        )
    ]

    for test in data_structure_tests:
        datasets.add_test_case(test)

    # Algorithm patterns
    algorithm_tests = [
        GoldenTestCase(
            id="binary_search",
            category="algorithms",
            input_code="""def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
            expected_explanation="This implements the binary search algorithm to find a target value in a sorted array. It uses two pointers (left and right) to define the search range, calculates the middle index, and compares the middle element with the target. Based on the comparison, it narrows the search range by half in each iteration. Returns the index if found, -1 otherwise. Time complexity: O(log n).",
            strategy="vanilla"
        ),
        GoldenTestCase(
            id="bubble_sort",
            category="algorithms",
            input_code="""def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
            expected_explanation="This implements the bubble sort algorithm to sort an array in ascending order. It uses nested loops to compare adjacent elements and swaps them if they're in the wrong order. The outer loop runs n times, and the inner loop runs fewer times each iteration as the largest elements 'bubble up' to the end. Time complexity: O(n²).",
            strategy="vanilla"
        )
    ]

    for test in algorithm_tests:
        datasets.add_test_case(test)

    # Error handling patterns
    error_handling_tests = [
        GoldenTestCase(
            id="try_except_division",
            category="error_handling",
            input_code="""def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return "Cannot divide by zero"
    except TypeError:
        return "Invalid input types" """,
            expected_explanation="This function performs safe division with error handling. It attempts to divide 'a' by 'b' within a try block. If a ZeroDivisionError occurs (division by zero), it returns an error message. If a TypeError occurs (invalid input types), it returns a different error message. This prevents the program from crashing on invalid inputs.",
            strategy="vanilla"
        )
    ]

    for test in error_handling_tests:
        datasets.add_test_case(test)

    # Object-oriented programming
    oop_tests = [
        GoldenTestCase(
            id="simple_class_definition",
            category="oop",
            input_code="""class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, my name is {self.name}"

    def birthday(self):
        self.age += 1""",
            expected_explanation="This defines a Person class with attributes 'name' and 'age'. The __init__ method is the constructor that initializes these attributes when a new Person object is created. The greet() method returns a greeting message with the person's name. The birthday() method increments the person's age by 1, simulating a birthday.",
            strategy="vanilla"
        )
    ]

    for test in oop_tests:
        datasets.add_test_case(test)

    return datasets


class GoldenTestRunner:
    """Runs golden tests and checks for regressions."""

    def __init__(self, datasets: GoldenTestDatasets):
        """Initialize the test runner.

        Args:
            datasets: Golden test datasets to run
        """
        self.datasets = datasets
        self.results: Dict[str, Any] = {}

    def run_tests(
        self,
        model_explainer,
        categories: Optional[List[str]] = None,
        tolerance: float = 0.8
    ) -> Dict[str, Any]:
        """Run golden tests against a model.

        Args:
            model_explainer: The code explainer model to test
            categories: Specific categories to test (None for all)
            tolerance: Similarity tolerance for pass/fail (0-1)

        Returns:
            Dictionary with test results
        """
        if categories is None:
            categories = self.datasets.get_all_categories()

        results = {
            "timestamp": time.time(),
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "categories": {},
            "failures": []
        }

        for category in categories:
            test_cases = self.datasets.get_category_tests(category)
            category_results = {
                "total": len(test_cases),
                "passed": 0,
                "failed": 0,
                "tests": []
            }

            for test_case in test_cases:
                try:
                    # Get explanation from model
                    actual_explanation = model_explainer.explain_code(
                        test_case.input_code,
                        strategy=test_case.strategy
                    )

                    # Check similarity (simplified - could use more sophisticated metrics)
                    similarity = self._calculate_similarity(
                        test_case.expected_explanation,
                        actual_explanation
                    )

                    passed = similarity >= tolerance

                    test_result = {
                        "id": test_case.id,
                        "passed": passed,
                        "similarity": similarity,
                        "expected": test_case.expected_explanation,
                        "actual": actual_explanation,
                        "input_hash": test_case.get_input_hash()
                    }

                    category_results["tests"].append(test_result)

                    if passed:
                        category_results["passed"] += 1
                        results["passed"] += 1
                    else:
                        category_results["failed"] += 1
                        results["failed"] += 1
                        results["failures"].append({
                            "category": category,
                            "test_id": test_case.id,
                            "similarity": similarity,
                            "reason": f"Similarity {similarity:.2f} below tolerance {tolerance}"
                        })

                    results["total_tests"] += 1

                except Exception as e:
                    logger.error(f"Error running test {test_case.id}: {e}")
                    category_results["failed"] += 1
                    results["failed"] += 1
                    results["total_tests"] += 1
                    results["failures"].append({
                        "category": category,
                        "test_id": test_case.id,
                        "error": str(e)
                    })

            results["categories"][category] = category_results

        self.results = results
        return results

    def _calculate_similarity(self, expected: str, actual: str) -> float:
        """Calculate similarity between expected and actual explanations.

        This is a simplified implementation. In practice, you might want to use
        more sophisticated metrics like BLEU, ROUGE, or semantic similarity.
        """
        # Simple word overlap similarity
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        if not expected_words:
            return 1.0 if not actual_words else 0.0

        intersection = expected_words.intersection(actual_words)
        union = expected_words.union(actual_words)

        return len(intersection) / len(union) if union else 0.0

    def generate_report(self, output_path: str = "./golden_test_report.md") -> None:
        """Generate a markdown report of test results."""
        if not self.results:
            logger.warning("No test results available for report generation")
            return

        results = self.results
        report_lines = [
            "# Golden Test Results",
            "",
            f"**Test Run:** {time.ctime(results['timestamp'])}",
            f"**Total Tests:** {results['total_tests']}",
            f"**Passed:** {results['passed']} ({results['passed']/results['total_tests']*100:.1f}%)",
            f"**Failed:** {results['failed']} ({results['failed']/results['total_tests']*100:.1f}%)",
            "",
            "## Category Summary",
            "",
            "| Category | Total | Passed | Failed | Pass Rate |",
            "|----------|--------|--------|--------|-----------|"
        ]

        for category, cat_results in results["categories"].items():
            total = cat_results["total"]
            passed = cat_results["passed"]
            pass_rate = (passed / total * 100) if total > 0 else 0
            report_lines.append(
                f"| {category} | {total} | {passed} | {cat_results['failed']} | {pass_rate:.1f}% |"
            )

        if results["failures"]:
            report_lines.extend([
                "",
                "## Failures",
                ""
            ])

            for failure in results["failures"]:
                report_lines.append(f"- **{failure['category']}/{failure['test_id']}:** {failure.get('reason', failure.get('error', 'Unknown error'))}")

        # Write report
        with open(output_path, 'w') as f:
            f.write("\n".join(report_lines))

        logger.info(f"Golden test report saved to: {output_path}")


def main():
    """Example usage of golden test system."""
    # Create golden test datasets
    datasets = create_core_golden_tests()
    datasets.save_datasets()

    print(f"Created {datasets.get_total_test_count()} golden tests across {len(datasets.get_all_categories())} categories")

    for category in datasets.get_all_categories():
        tests = datasets.get_category_tests(category)
        print(f"  {category}: {len(tests)} tests")


if __name__ == "__main__":
    main()
