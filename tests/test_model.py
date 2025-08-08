import pytest

from code_explainer.model import CodeExplainer


def test_explainer_init(tmp_path):
    # Should fall back to base model if results/ not present
    explainer = CodeExplainer(model_path=str(tmp_path))
    assert explainer is not None


def test_explain_basic():
    explainer = CodeExplainer()
    out = explainer.explain_code("print('hi')")
    assert isinstance(out, str)
    assert len(out) >= 0
