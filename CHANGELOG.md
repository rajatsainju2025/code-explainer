# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI: GitHub Actions with lint, type-check, tests, and coverage upload
- Pre-commit hooks (black, isort, ruff, mypy, basic checks)
- Testing: pytest unit tests and coverage
- Benchmarks: simple inference timing
- API: FastAPI server for explanation endpoint
- Web: Streamlit app scaffold
- Data: loaders (JSON/CSV/HF), augmentation utilities
- Metrics: BLEU, ROUGE-L, CodeBERTScore (optional)
- Dockerfile for web serving

### Changed
- Trainer now supports augmentation, eval metrics, and safer typing
- Config extended for data and optimization flags
- Requirements include metrics and API/web deps

### Fixed
- Tokenizer/model typing and safe saves

## [0.2.2] - 2025-08-09

### Added
- Colab quickstart notebook and README badge
- Discussions enabled with a “Start here” thread
- 10-day contribution plan and README section
- CODE_OF_CONDUCT.md, SECURITY.md, FUNDING.yml

### Changed
- Tests now use tiny models for speed
- pyproject/setup metadata updated for PyPI scaffold
- Package __version__ bumped to 0.2.2

## [0.2.1] - 2025-08-08

### Added
- Model presets: CodeT5 (small/base), CodeGPT-small, StarCoderBase-1B
- Eval CLI: `code-explainer eval` computing BLEU/ROUGE/BERTScore
- CLI aliases: `cx-train`, `cx-explain`, `cx-explain-file`, `cx-serve`
- Issue templates: Good First Issue and Roadmap
- Tests: eval CLI smoke test

### Changed
- Trainer: arch-aware (causal/seq2seq), dtype/8-bit config, proper tokenization/labels, seq2seq collator & trainer args
- Inference/app: config-driven `CodeExplainer` and refactored `app.py`, `train.py`
- README: alias examples and config-driven flows

## [0.2.0] - 2025-08-08

### Added
- Phase 1 foundation: code quality tooling, CI, tests, datasets/metrics modules, API and Streamlit scaffolds

### Changed
- Bump package version to 0.2.0

## [0.1.0] - 2025-01-XX

### Added
- Initial release
- Basic code explanation functionality
- Simple Gradio web interface
- DistilGPT-2 based model training
- Command line interface
- Basic configuration system

### Security
- Added input validation for code snippets
- Implemented safe model loading procedures

## [0.3.0] - 2025-08-12

### Added
- Prompt strategies: ast_augmented, retrieval_augmented, execution_trace
- CLI/API support for selecting prompt strategies
- Metrics: CodeBLEU (fallback to BLEU if missing)
- A/B strategy script `scripts/ab_compare_strategies.py`
- Docs: strategies, API usage, evaluation; examples for strategies
- API: `/strategies` endpoint and CORS + env-config
- Tooling: Makefile targets (api, eval-fast, ab), PR template, shellcheck hook
- Tests: prompt strategy unit tests

### Changed
- README updated with presets and strategy usage
- Version bumped to 0.3.0
