from code_explainer.utils import prompt_for_language

BASE_CFG = {
    "prompt": {
        "template": "Explain this Python code:\n```python\n{code}\n```\nExplanation:",
        "language_templates": {
            "python": "Explain this Python code:\n```python\n{code}\n```\nExplanation:"
        },
        "strategy": "vanilla",
    }
}


def test_ast_augmented_adds_context():
    code = '''\
import math

def add(a, b):
    """Add two numbers"""
    return a + b
'''
    cfg = {**BASE_CFG, "prompt": {**BASE_CFG["prompt"], "strategy": "ast_augmented"}}
    prompt = prompt_for_language(cfg, code)
    assert "Context:" in prompt


def test_retrieval_augmented_includes_docs():
    code = '''\
import math

def circle_area(r):
    """Compute area using math.pi"""
    return math.pi * r * r
'''
    cfg = {**BASE_CFG, "prompt": {**BASE_CFG["prompt"], "strategy": "retrieval_augmented"}}
    prompt = prompt_for_language(cfg, code)
    assert ("Own docstrings:" in prompt) or ("Imports docs:" in prompt)


def test_execution_trace_includes_outputs():
    code = "print(1+2)"
    cfg = {**BASE_CFG, "prompt": {**BASE_CFG["prompt"], "strategy": "execution_trace"}}
    prompt = prompt_for_language(cfg, code)
    assert "Execution trace" in prompt
