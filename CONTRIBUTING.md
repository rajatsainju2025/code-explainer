# Contributing to Code Explainer

We love your input! We want to make contributing to Code Explainer as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issues](https://github.com/rajatsainju2025/code-explainer/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/rajatsainju2025/code-explainer/issues/new).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install in development mode: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run these before submitting:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Testing

We use pytest for testing. Run tests with:

```bash
pytest
pytest --cov=code_explainer  # with coverage
```

## Documentation

We use docstrings for API documentation. Please follow the Google style:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
```

## Commit Messages

Please use clear and meaningful commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

# Contributing

We welcome contributions of all kinds!

## Getting Started
- Fork the repo and create a feature branch
- Run tests and linters locally
- Open a PR with a clear description

## Good First Issues
Use the template at `.github/ISSUE_TEMPLATE/good_first_issue.md` to create beginner-friendly tasks. Please:
- Add `good first issue` and `help wanted` labels
- Provide acceptance criteria and file pointers

## Roadmap Proposals
Use `.github/ISSUE_TEMPLATE/roadmap.md` for larger proposals. Include milestones and risks.

## Local Dev
- Install dev deps: `pip install -e .[dev,web]`
- Run tests: `pytest`
- Lint/typecheck: `flake8` / `mypy src/`

## Code Style
- Black + isort
- Type hints encouraged

Thanks for contributing!

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
