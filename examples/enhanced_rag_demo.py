#!/usr/bin/env python3
"""
Enhanced RAG Demo: Building a code knowledge base and using it for explanations.

This script demonstrates:
1. Building a FAISS index from a collection of code snippets
2. Using Enhanced RAG to explain code with contextual examples
3. Comparing explanations with and without RAG
"""

import json
import tempfile
from pathlib import Path

from code_explainer.model import CodeExplainer
from code_explainer.retrieval import CodeRetriever
from code_explainer.utils import load_config

# Sample code corpus for demonstration
SAMPLE_CODE_CORPUS = [
    # Sorting algorithms
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
    # Data structures
    "class Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        if not self.is_empty():\n            return self.items.pop()\n    \n    def is_empty(self):\n        return len(self.items) == 0",
    "class Queue:\n    def __init__(self):\n        self.items = []\n    \n    def enqueue(self, item):\n        self.items.insert(0, item)\n    \n    def dequeue(self):\n        if not self.is_empty():\n            return self.items.pop()\n    \n    def is_empty(self):\n        return len(self.items) == 0",
    # Search algorithms
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "def linear_search(arr, target):\n    for i, item in enumerate(arr):\n        if item == target:\n            return i\n    return -1",
    # Mathematical functions
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    # String operations
    "def reverse_string(s):\n    return s[::-1]",
    "def is_palindrome(s):\n    return s == s[::-1]",
    "def count_vowels(s):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in s if char in vowels)",
]

# Test queries to demonstrate RAG
TEST_QUERIES = [
    {
        "code": "def selection_sort(arr):\n    for i in range(len(arr)):\n        min_idx = i\n        for j in range(i+1, len(arr)):\n            if arr[j] < arr[min_idx]:\n                min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n    return arr",
        "description": "Selection sort algorithm (should retrieve other sorting algorithms)",
    },
    {
        "code": "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None",
        "description": "Binary tree class (should retrieve other data structures)",
    },
    {
        "code": "def power(base, exp):\n    if exp == 0:\n        return 1\n    return base * power(base, exp - 1)",
        "description": "Recursive power function (should retrieve other recursive functions)",
    },
]


def create_demo_config(index_path: str) -> dict:
    """Create a configuration for the demo."""
    return {
        "model": {
            "name": "microsoft/DialoGPT-small",  # Use small model for demo
            "max_length": 512,
        },
        "prompt": {
            "strategy": "enhanced_rag",
            "template": "Explain the following Python code:\n{code}",
        },
        "retrieval": {"index_path": index_path, "similarity_top_k": 3, "similarity_threshold": 0.5},
        "generation": {"max_new_tokens": 150, "temperature": 0.7},
    }


def demo_index_building():
    """Demonstrate building a FAISS index from code snippets."""
    print("🔨 Building FAISS index from sample code corpus...")
    print(f"   Corpus size: {len(SAMPLE_CODE_CORPUS)} code snippets")

    retriever = CodeRetriever()

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "demo_index.faiss"

        # Build the index
        retriever.build_index(SAMPLE_CODE_CORPUS, save_path=str(index_path))
        print(f"✅ Index built successfully and saved to {index_path}")

        # Test retrieval
        print("\n🔍 Testing code retrieval...")
        query = "def insertion_sort(arr): pass"
        similar_codes = retriever.retrieve_similar_code(query, k=3)

        print(f"Query: {query}")
        print(f"Retrieved {len(similar_codes)} similar code snippets:")
        for i, code in enumerate(similar_codes, 1):
            print(f"  {i}. {code[:60]}...")

        return str(index_path)


def demo_enhanced_rag_explanation(index_path: str):
    """Demonstrate Enhanced RAG explanations."""
    print("\n🤖 Demonstrating Enhanced RAG explanations...")

    # Create configuration
    config = create_demo_config(index_path)

    # For demo purposes, we'll simulate explanations since we need a trained model
    print("\n📝 Simulating Enhanced RAG explanations:")

    retriever = CodeRetriever()
    retriever.load_index(index_path)

    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"Query Code:\n{test_case['code']}")

        # Show retrieved similar codes
        similar_codes = retriever.retrieve_similar_code(test_case["code"], k=3)
        print(f"\nRetrieved {len(similar_codes)} similar examples:")
        for j, code in enumerate(similar_codes, 1):
            print(f"  Example {j}: {code[:80]}...")

        # Simulate enhanced prompt
        rag_context = [
            "You are a helpful assistant that uses the following similar code examples to provide a better explanation.",
            "---",
            "Similar Code Examples:",
        ]
        for j, example in enumerate(similar_codes):
            rag_context.append(f"\nExample {j+1}:\n```python\n{example}\n```")

        rag_context.append("---")
        rag_context.append(f"Explain the following Python code:\n{test_case['code']}")

        enhanced_prompt = "\n".join(rag_context)
        print(f"\nEnhanced RAG Prompt (truncated):\n{enhanced_prompt[:300]}...")


def demo_performance_comparison():
    """Demonstrate performance comparison between strategies."""
    print("\n📊 Strategy Performance Comparison")
    print("=" * 50)

    strategies = ["vanilla", "ast_augmented", "enhanced_rag"]
    sample_code = "def heap_sort(arr):\n    # Implementation here\n    pass"

    for strategy in strategies:
        print(f"\n🔧 Strategy: {strategy}")
        if strategy == "enhanced_rag":
            print("   ✅ Includes similar code examples for context")
            print("   ✅ Better domain-specific explanations")
            print("   ⚠️  Requires pre-built index")
            print("   ⚠️  Slight latency increase (~100-200ms)")
        elif strategy == "ast_augmented":
            print("   ✅ Includes code structure analysis")
            print("   ⚠️  Limited to syntax understanding")
        else:  # vanilla
            print("   ✅ Fast and simple")
            print("   ⚠️  Limited contextual understanding")


def main():
    """Run the Enhanced RAG demonstration."""
    print("🚀 Enhanced RAG Demo")
    print("=" * 50)

    try:
        # Step 1: Build index
        index_path = demo_index_building()

        # Step 2: Demonstrate RAG explanations
        demo_enhanced_rag_explanation(index_path)

        # Step 3: Performance comparison
        demo_performance_comparison()

        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Build an index from your own code corpus")
        print("2. Train a model with your domain-specific data")
        print("3. Use Enhanced RAG for better code explanations")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
