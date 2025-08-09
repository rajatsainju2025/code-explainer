import json
from pathlib import Path

from click.testing import CliRunner
from code_explainer.cli import main as cli_main

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


def test_cli_explain_happy_path(tmp_path: Path):
    cfg = _write_min_cfg(tmp_path)
    runner = CliRunner()
    # Pass a code snippet directly to avoid interactive mode
    result = runner.invoke(
        cli_main,
        [
            "explain",
            "--config",
            str(cfg),
            "print('hello')",
        ],
    )
    assert result.exit_code == 0, result.output
    # Should include the Explanation panel title
    assert "Explanation" in result.output


def test_cli_explain_file_happy_path(tmp_path: Path):
    cfg = _write_min_cfg(tmp_path)
    py = tmp_path / "snippet.py"
    py.write_text("x = [i*i for i in range(3)]\nprint(x)\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "explain-file",
            "--config",
            str(cfg),
            str(py),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Analysis" in result.output
