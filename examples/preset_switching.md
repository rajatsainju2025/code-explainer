# Quick Guide: Switching Model Presets

Use the preset YAMLs in `configs/` to switch architectures quickly.

Examples:

- DistilGPT-2 (default, causal):
  - Train: `cx-train -c configs/default.yaml`
  - Evaluate: `code-explainer eval -c configs/default.yaml`

- CodeT5 Small (seq2seq):
  - Train: `cx-train -c configs/codet5-small.yaml`
  - Evaluate: `code-explainer eval -c configs/codet5-small.yaml`

- CodeT5 Base (seq2seq):
  - Train: `cx-train -c configs/codet5-base.yaml`
  - Evaluate: `code-explainer eval -c configs/codet5-base.yaml`

- StarCoderBase 1B (causal):
  - Train: `cx-train -c configs/starcoderbase-1b.yaml`
  - Evaluate: `code-explainer eval -c configs/starcoderbase-1b.yaml`

Override paths via CLI flags (e.g., `--data` or `--test-file`).
