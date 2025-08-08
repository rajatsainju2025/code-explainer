# Examples

This folder contains quick-start examples to train, evaluate, and serve the Code Explainer with different presets.

## Data

Tiny example datasets are provided in `data/`:
- `data/train.json`
- `data/eval.json`
- `data/test.json`

## Train

DistilGPT-2 (default):

```bash
cx-train --config configs/default.yaml
```

CodeT5-Small:

```bash
cx-train --config configs/codet5-small.yaml
```

StarCoderBase-1B (8-bit recommended):

```bash
cx-train --config configs/starcoderbase-1b.yaml
```

## Evaluate

Evaluate a trained model using the configured `data.test_file`:

```bash
code-explainer eval --config configs/default.yaml
```

Or override test file:

```bash
code-explainer eval --config configs/codet5-small.yaml --test-file data/test.json
```

## Serve

Start a simple web UI:

```bash
cx-serve --port 7860
```

## Explain via CLI

Interactive:

```bash
code-explainer explain
```

Single snippet:

```bash
cx-explain "print('hello')"
```
