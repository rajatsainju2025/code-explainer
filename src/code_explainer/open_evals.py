"""Open evaluations module: datasets registry and runner.

This is a minimal scaffold to run standardized evaluations.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import random

logger = logging.getLogger(__name__)

# Registry of open-eval datasets
OPEN_EVAL_DATASETS = {
    "demo-addsub": {
        "name": "Demo Addition/Subtraction",
        "description": "Simple arithmetic problems for demonstration",
        "format": "qa",
        "size": 20,
        "generator": "generate_addsub_demo"
    },
    "demo-fizzbuzz": {
        "name": "Demo FizzBuzz Logic",
        "description": "FizzBuzz pattern recognition problems",
        "format": "qa",
        "size": 15,
        "generator": "generate_fizzbuzz_demo"
    },
    "code-understanding-basic": {
        "name": "Basic Code Understanding",
        "description": "Understanding simple code snippets and their functionality",
        "format": "qa",
        "size": 50,
        "generator": "generate_code_understanding_basic"
    },
    "code-debugging-common": {
        "name": "Common Code Debugging",
        "description": "Identifying and fixing common programming errors",
        "format": "qa",
        "size": 30,
        "generator": "generate_debugging_common"
    },
    "algorithm-complexity": {
        "name": "Algorithm Complexity Analysis",
        "description": "Analyzing time and space complexity of algorithms",
        "format": "qa",
        "size": 25,
        "generator": "generate_complexity_analysis"
    },
    "data-structures-usage": {
        "name": "Data Structures Usage",
        "description": "Appropriate usage of data structures for given problems",
        "format": "qa",
        "size": 40,
        "generator": "generate_data_structures"
    }
}


def generate_addsub_demo() -> List[Dict[str, Any]]:
    """Generate demo addition/subtraction problems."""
    problems = []
    for i in range(20):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(['+', '-'])
        if op == '+':
            answer = a + b
            question = f"What is {a} + {b}?"
        else:
            answer = a - b  
            question = f"What is {a} - {b}?"
        
        problems.append({
            "id": f"addsub_{i+1}",
            "question": question,
            "answer": str(answer),
            "metadata": {"operation": op, "difficulty": "easy"}
        })
    return problems


def generate_fizzbuzz_demo() -> List[Dict[str, Any]]:
    """Generate FizzBuzz pattern problems."""
    problems = []
    for i in range(15):
        num = random.randint(1, 100)
        expected = []
        if num % 3 == 0:
            expected.append("Fizz")
        if num % 5 == 0:
            expected.append("Buzz")
        if not expected:
            expected = [str(num)]
        
        answer = "".join(expected)
        question = f"What should be printed for the number {num} in FizzBuzz?"
        
        problems.append({
            "id": f"fizzbuzz_{i+1}",
            "question": question,
            "answer": answer,
            "metadata": {"number": num, "difficulty": "easy"}
        })
    return problems


def generate_code_understanding_basic() -> List[Dict[str, Any]]:
    """Generate basic code understanding problems."""
    problems = []
    
    code_snippets = [
        {
            "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "question": "What does this function compute?",
            "answer": "factorial of n"
        },
        {
            "code": "def is_palindrome(s):\n    return s == s[::-1]",
            "question": "What does this function check?",
            "answer": "if string is a palindrome"
        },
        {
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "question": "What sequence does this function generate?",
            "answer": "Fibonacci sequence"
        }
    ]
    
    for i, snippet in enumerate(code_snippets * 17):  # Repeat to get ~50
        if i >= 50:
            break
        problems.append({
            "id": f"code_understanding_{i+1}",
            "question": f"Code: ```python\n{snippet['code']}\n```\n{snippet['question']}",
            "answer": snippet['answer'],
            "metadata": {"type": "code_understanding", "difficulty": "basic"}
        })
    return problems


def generate_debugging_common() -> List[Dict[str, Any]]:
    """Generate common debugging problems."""
    problems = []
    
    bugs = [
        {
            "buggy_code": "def divide(a, b):\n    return a / b",
            "question": "What's the potential issue with this code?",
            "answer": "Division by zero error when b is 0"
        },
        {
            "buggy_code": "for i in range(len(arr)):\n    if arr[i] == target:\n        del arr[i]",
            "question": "What's wrong with this code?",
            "answer": "Modifying list while iterating causes index errors"
        },
        {
            "buggy_code": "def get_first(lst):\n    return lst[0]",
            "question": "What error can this function cause?",
            "answer": "IndexError if list is empty"
        }
    ]
    
    for i, bug in enumerate(bugs * 10):  # Repeat to get 30
        if i >= 30:
            break
        problems.append({
            "id": f"debugging_{i+1}",
            "question": f"Buggy code: ```python\n{bug['buggy_code']}\n```\n{bug['question']}",
            "answer": bug['answer'],
            "metadata": {"type": "debugging", "difficulty": "common"}
        })
    return problems


def generate_complexity_analysis() -> List[Dict[str, Any]]:
    """Generate algorithm complexity analysis problems."""
    problems = []
    
    algorithms = [
        {
            "code": "def linear_search(arr, target):\n    for i in range(len(arr)):\n        if arr[i] == target:\n            return i\n    return -1",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)"
        },
        {
            "code": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1-i):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]",
            "time_complexity": "O(n²)",
            "space_complexity": "O(1)"
        },
        {
            "code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1",
            "time_complexity": "O(log n)",
            "space_complexity": "O(1)"
        }
    ]
    
    for i, algo in enumerate(algorithms * 9):  # Repeat to get ~25
        if i >= 25:
            break
        question_type = random.choice(["time", "space"])
        if question_type == "time":
            question = f"What is the time complexity of this algorithm?\n```python\n{algo['code']}\n```"
            answer = algo['time_complexity']
        else:
            question = f"What is the space complexity of this algorithm?\n```python\n{algo['code']}\n```"
            answer = algo['space_complexity']
        
        problems.append({
            "id": f"complexity_{i+1}",
            "question": question,
            "answer": answer,
            "metadata": {"type": "complexity", "analysis_type": question_type}
        })
    return problems


def generate_data_structures() -> List[Dict[str, Any]]:
    """Generate data structures usage problems."""
    problems = []
    
    scenarios = [
        {
            "scenario": "You need to store items and retrieve them in Last-In-First-Out order",
            "answer": "Stack",
            "alternatives": ["Queue", "List", "Set"]
        },
        {
            "scenario": "You need to store items and retrieve them in First-In-First-Out order", 
            "answer": "Queue",
            "alternatives": ["Stack", "List", "Set"]
        },
        {
            "scenario": "You need to store unique items and check membership quickly",
            "answer": "Set",
            "alternatives": ["List", "Stack", "Queue"]
        },
        {
            "scenario": "You need to map keys to values for fast lookup",
            "answer": "Dictionary/HashMap",
            "alternatives": ["List", "Set", "Array"]
        }
    ]
    
    for i, scenario in enumerate(scenarios * 10):  # Repeat to get 40
        if i >= 40:
            break
        problems.append({
            "id": f"data_structures_{i+1}",
            "question": f"Which data structure is most appropriate for this scenario: {scenario['scenario']}",
            "answer": scenario['answer'],
            "metadata": {
                "type": "data_structures",
                "alternatives": scenario['alternatives']
            }
        })
    
    return problems


def get_dataset_list() -> List[str]:
    """Get list of available dataset IDs."""
    return list(OPEN_EVAL_DATASETS.keys())


def get_dataset_info(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific dataset."""
    return OPEN_EVAL_DATASETS.get(dataset_id)


def generate_dataset(dataset_id: str) -> List[Dict[str, Any]]:
    """Generate a dataset by calling its generator function."""
    if dataset_id not in OPEN_EVAL_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    
    generator_name = OPEN_EVAL_DATASETS[dataset_id]["generator"]
    generator_func = globals().get(generator_name)
    
    if generator_func is None:
        raise ValueError(f"Generator function not found: {generator_name}")
    
    return generator_func()


def run_eval(
    dataset_id: str,
    model_path: str = "./results",
    config_path: str = "configs/default.yaml",
    out_csv: Optional[str] = None,
    out_json: Optional[str] = None
) -> Dict[str, Any]:
    """Run evaluation on a specific dataset."""
    logger.info(f"Running evaluation on dataset: {dataset_id}")
    
    # Generate dataset
    dataset = generate_dataset(dataset_id)
    dataset_info = get_dataset_info(dataset_id)
    
    # Run evaluation (simplified for now)
    results = []
    correct = 0
    total = len(dataset)
    
    for sample in dataset:
        # Simulate model prediction (replace with actual model call)
        predicted = "mock_prediction"
        is_correct = predicted.lower().strip() == sample["answer"].lower().strip()
        
        if is_correct:
            correct += 1
        
        result = {
            "id": sample["id"],
            "question": sample["question"],
            "expected": sample["answer"],
            "predicted": predicted,
            "correct": is_correct,
            "metadata": sample.get("metadata", {})
        }
        results.append(result)
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "dataset": dataset_id,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "timestamp": time.time()
    }
    
    # Save outputs if requested
    if out_csv:
        save_results_csv(results, out_csv)
    if out_json:
        save_results_json({"metrics": metrics, "results": results}, out_json)
    
    logger.info(f"Evaluation completed: {correct}/{total} correct ({accuracy:.2%})")
    return metrics


def save_results_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """Save evaluation results to CSV."""
    if not results:
        return
    
    fieldnames = ["id", "question", "expected", "predicted", "correct"]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {k: result.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    logger.info(f"Results saved to CSV: {filepath}")


def save_results_json(data: Dict[str, Any], filepath: str) -> None:
    """Save evaluation data to JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to JSON: {filepath}")