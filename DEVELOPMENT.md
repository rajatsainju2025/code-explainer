# Development Guide

Complete guide for contributing to Code Explainer development.

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/code-explainer.git
cd code-explainer

# Add upstream remote
git remote add upstream https://github.com/rajatsainju2025/code-explainer.git
```

### 2. Install Development Dependencies

```bash
# Using Make (recommended)
make install-dev

# Or with Poetry
poetry install --all-extras --with dev

# Or with pip
pip install -e .[all]
pip install -r requirements-dev.txt
```

### 3. Setup Pre-commit Hooks

```bash
pre-commit install
```

## Development Workflow

### Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Make Changes

1. Write code following style guidelines
2. Add tests for new functionality
3. Update documentation
4. Run quality checks

### Run Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type

# Run tests
make test

# All quality checks
make quality
```

### Commit Changes

```bash
git add .
git commit -m "feat: your feature description"
```

Use conventional commit messages:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `test:` tests
- `refactor:` code refactoring
- `perf:` performance improvement
- `chore:` maintenance

### Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Testing

### Run All Tests

```bash
make test-all
```

### Run Specific Tests

```bash
# Unit tests only
make test-unit

# Integration tests
make test-integration

# With coverage
make test-cov
```

### Write Tests

Place tests in `tests/` directory:

```python
# tests/test_my_feature.py
import pytest
from code_explainer import CodeExplainer

def test_my_feature():
    explainer = CodeExplainer()
    result = explainer.some_method()
    assert result is not None
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for public APIs

Example:

```python
def explain_code(
    self,
    code: str,
    strategy: str = "vanilla",
    max_length: Optional[int] = None
) -> str:
    """Explain a code snippet.
    
    Args:
        code: Source code to explain
        strategy: Explanation strategy to use
        max_length: Maximum explanation length
        
    Returns:
        Generated explanation text
        
    Raises:
        ValueError: If code is empty
    """
    if not code.strip():
        raise ValueError("Code cannot be empty")
    # ...
```

### Formatting Tools

```bash
# Format with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check with flake8
flake8 src/ tests/
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """Short description.
    
    Longer description explaining the function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is invalid
        
    Example:
        >>> my_function("test", 42)
        True
    """
```

### Build Documentation

```bash
make docs-build
make docs-serve  # View at http://localhost:8000
```

## Debugging

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
python -m code_explainer.cli explain "your code"
```

### Use Python Debugger

```python
import pdb; pdb.set_trace()  # Add breakpoint
```

### VS Code Debugging

Add to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug API",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "code_explainer.api.main:app",
        "--reload"
      ],
      "env": {
        "LOG_LEVEL": "DEBUG"
      }
    }
  ]
}
```

## Performance Profiling

### CPU Profiling

```bash
python -m cProfile -o profile.stats your_script.py
python -m pstats profile.stats
```

### Memory Profiling

```bash
pip install memory-profiler
python -m memory_profiler your_script.py
```

### Benchmark Changes

```bash
make benchmark-baseline  # Before changes
# Make your changes
make benchmark-compare   # Compare performance
```

## Database Migrations

If adding database changes:

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head
```

## Release Process

### 1. Update Version

```bash
# Update version in pyproject.toml and __init__.py
make bump-minor  # or bump-major, bump-patch
```

### 2. Update Changelog

Edit `CHANGELOG.md` with new version changes.

### 3. Create Release

```bash
git tag v0.x.0
git push --tags
make release
```

## Troubleshooting

### Import Errors

```bash
# Reinstall in editable mode
pip install -e .
```

### Test Failures

```bash
# Run with verbose output
pytest -v -s tests/

# Run specific test
pytest tests/test_my_feature.py::test_specific_function
```

### Build Errors

```bash
# Clean and rebuild
make clean
make install-dev
```

## Best Practices

1. **Small PRs**: Keep pull requests focused and small
2. **Test Coverage**: Aim for >80% coverage
3. **Documentation**: Update docs with code changes
4. **Type Hints**: Add type hints to new code
5. **Performance**: Profile before optimizing
6. **Security**: Run security scans before release

## Resources

- [Python Style Guide (PEP 8)](https://peps.python.org/pep-0008/)
- [Type Hints (PEP 484)](https://peps.python.org/pep-0484/)
- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Getting Help

- Ask in [GitHub Discussions](https://github.com/rajatsainju2025/code-explainer/discussions)
- Check existing [Issues](https://github.com/rajatsainju2025/code-explainer/issues)
- Review [Contributing Guidelines](CONTRIBUTING.md)
