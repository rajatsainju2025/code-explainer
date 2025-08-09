from code_explainer.utils import prompt_for_language


def test_prompt_for_language_overrides():
    cfg = {
        "prompt": {
            "template": "Explain this code:\n{code}\nExplanation:",
            "language_templates": {
                "javascript": "Explain this JavaScript code:\n{code}\nExplanation:",
                "cpp": "Explain this C++ code:\n{code}\nExplanation:",
            },
        }
    }
    js = "console.log('hi')"
    out_js = prompt_for_language(cfg, js)
    assert out_js.startswith("Explain this JavaScript code:"), out_js

    py = "print('hi')"
    out_py = prompt_for_language(cfg, py)
    assert out_py.startswith("Explain this code:"), out_py
