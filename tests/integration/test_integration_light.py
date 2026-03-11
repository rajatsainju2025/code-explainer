"""Lightweight integration tests using mocks to avoid model downloads.

These tests validate high-level flows: explanation generation, symbolic combination,
and retrieval flow using small, fast mocks.
"""

from unittest.mock import MagicMock, patch
import torch

import pytest

from code_explainer.model import CodeExplainer
from code_explainer.retrieval.retriever import CodeRetriever


def make_dummy_explainer():
    # Create an explainer with mocked model and tokenizer to avoid heavy deps
    expl = CodeExplainer(config_path="configs/codet5-small.yaml")

    # Inject simple tokenizer/model mocks
    tokenizer = MagicMock()
    # Prefer setting the mock's return_value so the mock is callable in all contexts
    # Use small real tensors so downstream code that inspects shapes works
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
    }
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "PROMPT Generated explanation"

    model = MagicMock()
    # Return a simple tensor-like object for generate
    model.generate.return_value = [[1, 2, 3, 4, 5]]

    expl.tokenizer = tokenizer
    expl.model = model

    return expl


def test_explain_code_basic():
    expl = make_dummy_explainer()

    code = "def add(a, b):\n    return a + b"
    text = expl.explain_code(code)

    assert isinstance(text, str)
    assert len(text) > 0


def test_explain_code_with_symbolic_integration():
    expl = make_dummy_explainer()

    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    combined = expl.explain_code_with_symbolic(code, include_symbolic=True)
    assert "## Symbolic Analysis" in combined
    assert "## Code Explanation" in combined


def test_retriever_initialization_with_mock():
    # Ensure CodeRetriever can be constructed with a mock model
    mock_model = MagicMock()
    retr = CodeRetriever(model_name="mock-model", model=mock_model)
    assert hasattr(retr, "retrieve_similar_code")

