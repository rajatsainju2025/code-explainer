import json
from pathlib import Path

from code_explainer.cli import main as cli_main
from click.testing import CliRunner


def test_cli_eval_smoke(tmp_path: Path, monkeypatch):
    # Create a tiny test dataset
    test_data = [
        {"code": "print('hi')", "explanation": "Prints hi"},
        {"code": "x=1+1", "explanation": "Adds numbers"},
    ]
    test_file = tmp_path / "test.json"
    test_file.write_text(json.dumps(test_data))

    # Minimal config referencing the test file
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
model:
  arch: "causal"
  name: "distilgpt2"
  max_length: 64
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  torch_dtype: "auto"
  load_in_8bit: false
training:
  output_dir: "./results"
logging:
  level: "ERROR"
  log_file: "logs/test.log"
data:
  test_file: "%s"
        """ % str(test_file)
    )

    runner = CliRunner()
    result = runner.invoke(cli_main, ["eval", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "bleu" in result.output
