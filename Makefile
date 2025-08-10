# Simple developer Makefile
.PHONY: install test lint type format precommit clean

install:
	python -m pip install --upgrade pip
	pip install -e .[dev]

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/

type:
	mypy src/

precommit:
	pre-commit run --all-files

test:
	pytest --cov=code_explainer --cov-report=term-missing

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info coverage.xml
