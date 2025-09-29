# Enhanced Makefile for Code Explainer
# Supports both Poetry and pip workflows with device detection

.PHONY: help install install-poetry install-pip install-dev install-minimal install-full
.PHONY: test lint type format precommit clean validate-env
.PHONY: requirements generate-requirements
.PHONY: api eval-fast ab api-serve icml-all icml-analysis icml-outputs
.PHONY: docs-serve docs-deploy device-info

# Default target
help: ## Show this help message
	@echo "Code Explainer Development Makefile"
	@echo ""
	@echo "Installation Commands:"
	@echo "  install          Auto-detect and install (Poetry preferred)"
	@echo "  install-poetry   Install using Poetry with all dependencies"
	@echo "  install-pip      Install using pip with basic dependencies"
	@echo "  install-dev      Install development environment"
	@echo "  install-minimal  Install minimal dependencies only"
	@echo "  install-full     Install all optional features"
	@echo ""
	@echo "Development Commands:"
	@echo "  test            Run tests"
	@echo "  lint            Run linting"
	@echo "  format          Format code"
	@echo "  type            Run type checking"
	@echo "  validate-env    Validate development environment"
	@echo "  device-info     Show available compute devices"
	@echo ""
	@echo "Utility Commands:"
	@echo "  requirements    Generate requirements.txt from Poetry"
	@echo "  clean           Clean cache and build files"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Auto-detect installation method
install: ## Auto-detect and install dependencies (Poetry preferred)
	@if command -v poetry >/dev/null 2>&1; then \
		echo "🎯 Using Poetry for installation..."; \
		$(MAKE) install-poetry; \
	else \
		echo "📦 Poetry not found, using pip..."; \
		echo "💡 Consider installing Poetry for better dependency management"; \
		$(MAKE) install-pip; \
	fi

install-poetry: ## Install using Poetry with all dependencies
	@echo "🔧 Installing with Poetry..."
	@poetry --version || (echo "❌ Poetry not found. Install with: curl -sSL https://install.python-poetry.org | python3 -" && exit 1)
	poetry install --all-extras
	@echo "✅ Poetry installation complete!"
	@$(MAKE) validate-env

install-pip: ## Install using pip with basic dependencies
	@echo "🔧 Installing with pip..."
	python -m pip install --upgrade pip
	pip install -e .
	@echo "✅ Pip installation complete!"
	@echo "💡 For optional features, run: pip install -e .[web,rag,metrics]"
	@$(MAKE) validate-env

install-dev: ## Install development environment
	@if command -v poetry >/dev/null 2>&1; then \
		poetry install --all-extras --with dev; \
	else \
		pip install -e .[all] && pip install -r requirements-dev.txt 2>/dev/null || echo "⚠️  Dev requirements not found"; \
	fi
	pre-commit install 2>/dev/null || echo "⚠️  Pre-commit not available"
	@$(MAKE) validate-env

install-minimal: ## Install minimal dependencies only
	@if command -v poetry >/dev/null 2>&1; then \
		poetry install --only main; \
	else \
		pip install -r requirements-core.txt; \
	fi

install-full: ## Install all optional features
	@if command -v poetry >/dev/null 2>&1; then \
		poetry install --all-extras; \
	else \
		pip install -e .[all]; \
	fi

# Development commands
format: ## Format code with black and isort
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run black src/ tests/; \
		poetry run isort src/ tests/; \
	else \
		black src/ tests/; \
		isort src/ tests/; \
	fi

lint: ## Run linting
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run flake8 src/ tests/; \
	else \
		flake8 src/ tests/; \
	fi

type: ## Run type checking
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run mypy src/; \
	else \
		mypy src/; \
	fi

precommit: ## Run pre-commit hooks
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit run --all-files; \
	else \
		echo "⚠️  pre-commit not installed. Run: pip install pre-commit"; \
	fi

test: ## Run tests with coverage
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest --cov=code_explainer --cov-report=term-missing; \
	else \
		pytest --cov=code_explainer --cov-report=term-missing; \
	fi

# Utility commands
validate-env: ## Validate development environment
	@echo "🔍 Validating environment..."
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "⚠️  PyTorch not found"
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "⚠️  Transformers not found"
	@$(MAKE) device-info

device-info: ## Show available compute devices
	@echo "🖥️  Device Information:"
	@python -c "from src.code_explainer.device_manager import device_manager; import json; print(json.dumps(device_manager.get_device_info(), indent=2))" 2>/dev/null || python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {getattr(torch.backends, \"mps\", None) and torch.backends.mps.is_available()}')"

requirements: generate-requirements ## Generate requirements.txt from Poetry (alias)

generate-requirements: ## Generate requirements.txt from Poetry
	@if command -v poetry >/dev/null 2>&1; then \
		echo "📝 Generating requirements.txt from Poetry..."; \
		poetry export -f requirements.txt --output requirements.txt --without-hashes; \
		poetry export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes; \
		poetry export -f requirements.txt --output requirements-web.txt --extras web --without-hashes; \
		poetry export -f requirements.txt --output requirements-rag.txt --extras rag --without-hashes; \
		poetry export -f requirements.txt --output requirements-all.txt --all-extras --without-hashes; \
		echo "✅ Requirements files generated!"; \
	else \
		echo "⚠️  Poetry not found. Cannot generate requirements.txt"; \
	fi
.PHONY: eval-tiny
eval-tiny:
	code-explainer eval -c configs/default.yaml -t data/examples/tiny_eval.jsonl --self-consistency 2 --max-samples 2

clean: ## Clean cache and build files
	@echo "🧹 Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/ coverage.xml
	@echo "✅ Clean complete!"

api:
	uvicorn code_explainer.api.server:app --host 0.0.0.0 --port $${PORT:-8000} --reload

api-serve:
	bash scripts/serve_api.sh

eval-fast:
	code-explainer eval --config configs/default.yaml --max-samples 5

ab:
	python scripts/ab_compare_strategies.py --config configs/default.yaml --max-samples 5 --strategies vanilla ast_augmented

icml-all:
	python scripts/run_icml_experiments.py --config configs/icml_experiment_full.yaml --phase all

icml-analysis:
	python scripts/run_icml_experiments.py --config configs/icml_experiment_full.yaml --phase analysis

icml-outputs:
	python scripts/run_icml_experiments.py --config configs/icml_experiment_full.yaml --phase outputs

docs-serve:
	mkdocs serve -a 0.0.0.0:8001

docs-deploy:
	mkdocs gh-deploy --force

# Data governance helpers
.PHONY: intake-validate provenance-cards
intake-validate:
	python scripts/validate_intake.py data

provenance-cards:
	python scripts/provenance_card.py --preds out/preds.jsonl --out out/cards
