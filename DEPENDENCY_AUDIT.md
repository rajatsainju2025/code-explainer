# Dependency Audit Report

## Summary
- Total dependencies: 15+
- Critical dependencies: 5
- Development dependencies: 8+
- Unused packages: 0
- Outdated packages: 2 (recommend upgrade)

## Core Dependencies

### ML/NLP (Essential)
- **transformers** (4.34+): HuggingFace transformer models - CRITICAL
- **torch**: PyTorch for model inference - CRITICAL
- **pydantic** (2.0+): Data validation - CRITICAL

### Code Analysis (Essential)
- **ast**: Built-in Python AST parsing - CRITICAL
- **tree-sitter**: Code parsing - CRITICAL

### CLI & Web (Required)
- **click** (8.1+): CLI framework
- **streamlit** (1.28+): Web UI framework

### Utilities
- **python-dotenv**: Environment configuration
- **pyyaml**: YAML parsing
- **requests**: HTTP client

## Development Dependencies

### Testing & Quality
- **pytest** (7.4+): Test framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning
- **pre-commit**: Git hooks

## Optimization Recommendations

### Keep as-is (High value, actively maintained)
✅ transformers - Model inference backbone
✅ torch - ML framework
✅ pydantic - Type safety
✅ click - CLI UX
✅ pytest - Test infrastructure

### Review (Consider upgrade)
⚠️ streamlit (1.28 → 1.30+) - New features, improvements
⚠️ black (23.x → latest) - Better formatting rules

### Remove candidates
❌ None identified - all dependencies justified

## Import Analysis

**Top 10 Most Used Modules:**
1. `transformers.pipeline` - Used in core.py, model.py
2. `torch` - Used in training, inference
3. `pydantic` - Used in validation, config
4. `click` - Used in CLI commands
5. `logging` - Used throughout
6. `asyncio` - Used in async operations
7. `typing` - Type hints throughout
8. `pathlib` - File operations
9. `json` - Config and results serialization
10. `yaml` - Config file parsing

## Installation Size Analysis

Approximate sizes (with subdependencies):
- transformers + torch: ~2.5GB
- Click: ~1MB
- Pydantic: ~5MB
- streamlit: ~200MB
- pytest (dev): ~50MB

**Total production**: ~2.7GB
**Total development**: ~2.8GB

## Security Audit

✅ All dependencies from PyPI
✅ No known vulnerabilities (as of last check)
✅ Input validation enforced with pydantic
✅ Custom security module for code input

## Recommendations

1. **Keep current setup** - Well-justified dependencies
2. **Add to pre-commit**: `pip-audit` for vulnerability scanning
3. **Update streamlit** to 1.30+ for latest features
4. **Consider optional dependencies**: transformers[torch] syntax
5. **Document architecture** showing dependency relationships

## Dependencies by Module

### src/code_explainer/model/
- transformers, torch, pydantic, logging, typing

### src/code_explainer/cli_commands/
- click, pathlib, logging, typing, json

### src/code_explainer/trainer/
- torch, transformers, tqdm, logging, typing

### tests/
- pytest, unittest (built-in), json, typing
