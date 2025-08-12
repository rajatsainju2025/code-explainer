# Simple developer Makefile
.PHONY: install test lint type format precommit clean api eval-fast ab

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

api:
	uvicorn code_explainer.api.server:app --host 0.0.0.0 --port $${PORT:-8000} --reload

eval-fast:
	code-explainer eval --config configs/default.yaml --max-samples 5

ab:
	python scripts/ab_compare_strategies.py --config configs/default.yaml --max-samples 5 --strategies vanilla ast_augmented
