from pathlib import Path
from click.testing import CliRunner
import json

from code_explainer.cli import main as cli_main


def test_cli_eval_jsonl_with_provenance_and_sc(tmp_path: Path, monkeypatch):
    # Create a tiny JSONL dataset with provenance fields
    jsonl = tmp_path / "tiny.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({
                    "id": "1",
                    "code": "print('hi')",
                    "explanation": "Prints hi",
                    "source_ids": ["a", "b"],
                }),
                json.dumps({
                    "id": "2",
                    "code": "x=1+1",
                    "explanation": "Adds numbers",
                    "sources": ["x", "y"],
                }),
            ]
        )
    )

    # Minimal config
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
model:
  arch: "causal"
  name: "sshleifer/tiny-gpt2"
  max_length: 32
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  torch_dtype: "auto"
  load_in_8bit: false
data:
  test_file: """
        + str(jsonl)
    )

    # Monkeypatch explain_code to deterministic outputs with citations
    from code_explainer import model as model_mod

    def fake_explain_code(self, code, max_length=None, strategy=None):
        # Emit citations [a], [x] so we get non-zero provenance precision/recall
        if "hi" in code:
            return "Says hi [a]."
        return "Adds numbers [x]."

    monkeypatch.setattr(model_mod.CodeExplainer, "explain_code", fake_explain_code, raising=True)

    runner = CliRunner()
    result = runner.invoke(
        cli_main, ["eval", "--config", str(cfg), "--self-consistency", "2"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    # Check that provenance and self-consistency are mentioned in output
    assert "Prov. precision" in result.output or "provenance_precision" in result.output
    assert "Self-consistency" in result.output or "self_consistency_bleu" in result.output
