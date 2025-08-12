"""Tests for enhanced RAG retrieval functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from code_explainer.retrieval import CodeRetriever
from code_explainer.utils import prompt_for_language


class TestCodeRetriever:
    """Test cases for CodeRetriever class."""

    def test_code_retriever_init(self):
        """Test CodeRetriever initialization."""
        retriever = CodeRetriever()
        assert retriever.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert retriever.index is None
        assert retriever.code_corpus == []

    def test_code_retriever_custom_model(self):
        """Test CodeRetriever with custom model."""
        retriever = CodeRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert retriever.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_build_index(self):
        """Test building FAISS index from code snippets."""
        retriever = CodeRetriever()
        codes = [
            "def hello(): print('hello')",
            "def goodbye(): print('goodbye')",
            "class Person: pass",
            "import numpy as np",
        ]

        retriever.build_index(codes)
        assert retriever.index is not None
        assert len(retriever.code_corpus) == 4

    def test_save_and_load_index(self):
        """Test saving and loading FAISS index."""
        retriever = CodeRetriever()
        codes = ["def add(a, b): return a + b", "def multiply(a, b): return a * b"]

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test_index.faiss"

            # Build and save index
            retriever.build_index(codes, save_path=str(index_path))

            # Create new retriever and load index
            new_retriever = CodeRetriever()
            new_retriever.load_index(str(index_path))

            assert new_retriever.index is not None
            assert len(new_retriever.code_corpus) == 2

    def test_retrieve_similar_code(self):
        """Test retrieving similar code snippets."""
        retriever = CodeRetriever()
        codes = [
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b",
            "def multiply(x, y): return x * y",
            "class Calculator: pass",
            "import math",
        ]

        retriever.build_index(codes)

        # Query for math-related code
        query = "def divide(x, y): return x / y"
        similar_codes = retriever.retrieve_similar_code(query, k=3)

        assert len(similar_codes) <= 3
        assert isinstance(similar_codes, list)
        for code in similar_codes:
            assert isinstance(code, str)

    def test_retrieve_empty_corpus(self):
        """Test retrieving from empty corpus raises error."""
        retriever = CodeRetriever()
        query = "def test(): pass"

        with pytest.raises(ValueError, match="Index is not loaded or built"):
            retriever.retrieve_similar_code(query, k=3)

    def test_retrieve_no_index(self):
        """Test retrieving without building index raises error."""
        retriever = CodeRetriever()
        query = "def example(): pass"

        with pytest.raises(ValueError, match="Index is not loaded or built"):
            retriever.retrieve_similar_code(query, k=3)


class TestEnhancedRAGPrompt:
    """Test cases for enhanced RAG prompt strategy."""

    def test_enhanced_rag_prompt_no_index(self, tmp_path):
        """Test enhanced RAG prompt when no index is available."""
        config = {
            "prompt": {
                "strategy": "enhanced_rag",
                "template": "Explain the following Python code:\n{code}",
            },
            "retrieval": {"index_path": str(tmp_path / "nonexistent.faiss")},
        }

        code = "def test(): return 42"
        prompt = prompt_for_language(config, code)

        # Should fallback to vanilla prompt
        assert "similar code examples" not in prompt.lower()
        assert "explain" in prompt.lower()

    def test_enhanced_rag_prompt_with_index(self, tmp_path):
        """Test enhanced RAG prompt with available index."""
        # Create a simple index
        retriever = CodeRetriever()
        codes = [
            "def hello(): print('hello world')",
            "def add(a, b): return a + b",
            "class MyClass: pass",
        ]

        index_path = tmp_path / "test_index.faiss"
        retriever.build_index(codes, save_path=str(index_path))

        config = {
            "prompt": {
                "strategy": "enhanced_rag",
                "template": "Explain the following Python code:\n{code}",
            },
            "retrieval": {"index_path": str(index_path)},
        }

        code = "def greet(): print('hi there')"
        prompt = prompt_for_language(config, code)

        # Should include similar code examples
        assert "similar code examples" in prompt.lower()
        assert "example" in prompt.lower()

    def test_enhanced_rag_non_python(self):
        """Test enhanced RAG with non-Python code."""
        config = {
            "prompt": {
                "strategy": "enhanced_rag",
                "template": "Explain the following code:\n{code}",
            },
            "retrieval": {"index_path": "dummy.faiss"},
        }
        code = "console.log('hello');"

        prompt = prompt_for_language(config, code)

        # Should fallback to vanilla for non-Python
        assert "similar code examples" not in prompt.lower()


class TestRAGIntegration:
    """Integration tests for RAG functionality."""

    def test_end_to_end_rag_workflow(self, tmp_path):
        """Test complete RAG workflow from index building to retrieval."""
        # Create sample training data
        train_data = [
            {
                "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "explanation": "Recursive fibonacci",
            },
            {
                "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                "explanation": "Recursive factorial",
            },
            {
                "code": "def sum_list(lst):\n    return sum(lst)",
                "explanation": "Sum of list elements",
            },
        ]

        train_file = tmp_path / "train.json"
        with open(train_file, "w") as f:
            json.dump(train_data, f)

        # Build index
        retriever = CodeRetriever()
        codes = [item["code"] for item in train_data]
        index_path = tmp_path / "index.faiss"
        retriever.build_index(codes, save_path=str(index_path))

        # Test retrieval
        query = "def power(x, n):\n    if n == 0:\n        return 1\n    return x * power(x, n-1)"
        similar_codes = retriever.retrieve_similar_code(query, k=2)

        assert len(similar_codes) <= 2
        # Should retrieve recursive functions (fibonacci and factorial)
        assert any("fibonacci" in code or "factorial" in code for code in similar_codes)

    def test_rag_with_large_corpus(self):
        """Test RAG with larger code corpus."""
        retriever = CodeRetriever()

        # Generate diverse code snippets
        codes = []
        for i in range(50):
            if i % 3 == 0:
                codes.append(f"def func_{i}(x): return x + {i}")
            elif i % 3 == 1:
                codes.append(f"class Class_{i}: def __init__(self): self.value = {i}")
            else:
                codes.append(f"import module_{i}")

        retriever.build_index(codes)

        # Test retrieval
        query = "def new_function(y): return y + 100"
        similar_codes = retriever.retrieve_similar_code(query, k=5)

        assert len(similar_codes) <= 5
        # Should prefer function definitions
        function_codes = [code for code in similar_codes if "def " in code]
        assert len(function_codes) > 0


if __name__ == "__main__":
    pytest.main([__file__])
