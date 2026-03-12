"""End-to-end integration tests (mocked) for explanation + retrieval flows.

These tests exercise higher-level interactions while avoiding heavy model
downloads by using small in-memory mocks for tokenizer/model/embeddings.
"""

from unittest.mock import MagicMock

import torch

from code_explainer.model import CodeExplainer
from code_explainer.retrieval.retriever import CodeRetriever


def make_mock_explainer():
    expl = CodeExplainer(config_path="configs/codet5-small.yaml")
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = "PROMPT Generated explanation"

    mdl = MagicMock()
    mdl.generate.return_value = [[1, 2, 3, 4]]

    expl.tokenizer = tok
    expl.model = mdl
    return expl


def test_explain_and_retrieve_flow(tmp_path):
    expl = make_mock_explainer()

    # Build a small retrieval instance with a mock embedding model
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
    retr = CodeRetriever(model_name="mock-model", model=mock_model)

    # Build index from a tiny corpus
    codes = ["def add(a, b): return a + b", "def sub(a, b): return a - b"]
    retr.build_index(codes)

    # Ask explainer for explanation and ensure retrieval works
    code = codes[0]
    explanation = expl.explain_code_with_symbolic(code, include_symbolic=False)
    assert isinstance(explanation, str)

    results = retr.retrieve_similar_code(code, k=1)
    assert isinstance(results, list)
    assert len(results) == 1
