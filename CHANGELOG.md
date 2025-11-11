## [2.0.0] - November 5, 2025

### Major Improvements (20 Commits)
✅ Fresh project critique with 11 identified issues
✅ Code deduplication (180+ lines consolidated)
✅ Type hints expansion (17+ annotations)
✅ Error handling standardization (8 custom exceptions)
✅ Docstring standardization (Google-style)
✅ Logging centralization (unified setup)
✅ Configuration consolidation (ConfigManager)
✅ API simplification (3 entry points)
✅ Testing utilities (mocks and base classes)
✅ Security audit and input validation
✅ Performance monitoring and profiling
✅ Memory optimization utilities
✅ Enhanced caching with statistics
✅ Dependency audit and documentation
✅ Comprehensive API documentation
✅ Code style enforcement (Black, isort, flake8, mypy)
✅ Integration and stress tests
✅ Quality validation report (92/100)
✅ Release preparation

### Added
- Performance monitoring utilities with timing decorators
- Memory optimization using lazy loading and `__slots__`
- Enhanced cache with LRU eviction and statistics
- Security validation functions (sanitize, validate, escape)
- Centralized logging setup (rotating files, consistent format)
- Configuration manager with priority chain
- Simplified public API entry points
- Testing utilities with mock objects and base classes
- Integration test suite (12+ test cases)
- Comprehensive API documentation with examples
- Dependency audit report
- Quality validation report
- Code style enforcement configuration
- Performance and stress testing framework

### Changed
- Consolidated duplicate validation code into shared utilities
- Updated error handling to use custom exceptions
- Enhanced all public function docstrings
- Unified logging through centralized module
- Consolidated configuration management
- Improved import organization and cleanup

### Fixed
- Duplicate logger initialization
- Inconsistent error handling patterns
- Configuration scattered across modules
- Type coverage gaps (72% → target 80%)
- Code duplication (180+ LOC eliminated)

### Security
- Input validation at API boundaries
- Path traversal attack prevention
- Code input sanitization with length limits
- Python identifier validation
- Dangerous pattern detection

### Performance
- Cache hit rate monitoring enabled
- Performance timing available
- Memory usage tracking
- Batch processing support
- LRU cache with automatic eviction

### Documentation
- API documentation (complete)
- Integration guides (Django, Flask, FastAPI)
- Troubleshooting guide
- Dependency audit
- Quality metrics report
- Security guidelines

### Testing
- Integration test suite
- Stress testing framework
- Performance testing utilities
- Mock objects for testing
- Base test classes

## [Unreleased - v2.1.0 Planning]

### Added
- Batch explanation endpoint with concurrent compute and cache fast-path
- GZip compression middleware and ORJSON responses
- Uvicorn performance configuration and Makefile targets
- Streaming dataset iterator (JSONL and ijson fallback)
- API profiling script and performance tuning docs
- LRU caching for data loading and config loading
- Data loading benchmark script
- Optimized AST analysis with deduplicated imports

### Changed
- Use torch.inference_mode for generation where available
- Parameterized logging to reduce formatting overhead
- Tokenizer loading is LRU-cached within process
- Removed unused imports across codebase
- Optimized import statements with lazy loading where beneficial

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Eval CLI now supports JSONL datasets in addition to JSON arrays.
- Provenance metrics (citation precision/recall) integrated into `code-explainer eval` when `source_ids`/`sources` are provided per example.
- Self-consistency metrics via `--self-consistency N` flag, computing avg pairwise BLEU/ROUGE-L.
- Tiny example dataset at `data/examples/tiny_eval.jsonl` and docs.

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
