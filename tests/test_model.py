import json
import pytest
from pathlib import Path

from code_explainer.model import CodeExplainer

TINY_CAUSAL = "sshleifer/tiny-gpt2"


def _write_min_cfg(tmp_path: Path) -> Path:
    cfg = {
        "model": {
            "arch": "causal",
            "name": TINY_CAUSAL,
            "max_length": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "torch_dtype": "auto",
            "load_in_8bit": False,
        },
        "training": {"output_dir": str(tmp_path / "out")},
        "prompt": {"template": "Explain this Python code:\n{code}\nExplanation:"},
        "logging": {"level": "ERROR", "log_file": None},
        "data": {},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_explainer_init(tmp_path):
    cfg = _write_min_cfg(tmp_path)
    # Should fall back to base model if results/ not present
    explainer = CodeExplainer(model_path=str(tmp_path), config_path=str(cfg))
    assert explainer is not None


def test_explain_basic(tmp_path):
    cfg = _write_min_cfg(tmp_path)
    explainer = CodeExplainer(config_path=str(cfg))
    out = explainer.explain_code("print('hi')")
    assert isinstance(out, str)
    assert len(out) >= 0
