# Enhanced Makefile for Code Explainer
# Supports both Poetry and pip workflows with device detection

.PHONY: help install install-poetry install-pip install-dev install-minimal install-full
.PHONY: test test-all test-unit test-integration test-e2e test-cov
.PHONY: benchmark benchmark-regression benchmark-baseline benchmark-compare benchmark-profile benchmark-ci benchmark-report benchmark-all
.PHONY: security-scan security-vulnerabilities security-code security-semgrep security-all
.PHONY: cache-stats cache-clear cache-invalidate cache-warmup cache-backup cache-restore cache-all cache-test
.PHONY: lint type format precommit check quality setup
.PHONY: clean validate-env
.PHONY: requirements generate-requirements
.PHONY: api eval-fast ab api-serve icml-all icml-analysis icml-outputs
.PHONY: docs-serve docs-deploy docs-build
.PHONY: docker-build docker-run docker-dev
.PHONY: release version bump-major bump-minor bump-patch
.PHONY: device-info intake-validate provenance-cards

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
	@echo "  benchmark       Run performance benchmarks"
	@echo "  benchmark-all   Run complete benchmarking suite"
	@echo "  security-scan   Run comprehensive security scan"
	@echo "  security-all    Run complete security audit"
	@echo "  cache-stats     Show cache statistics"
	@echo "  cache-clear     Clear all cache entries"
	@echo "  cache-all       Run all cache management operations"
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

# Enhanced test targets
test-all: ## Run all test types (unit, integration, e2e)
	@echo "🧪 Running all tests..."
	$(MAKE) test-unit
	$(MAKE) test-integration
	$(MAKE) test-e2e

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest tests/unit/ -v; \
	else \
		pytest tests/unit/ -v; \
	fi

test-integration: ## Run integration tests only
	@echo "🔗 Running integration tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -m integration -v; \
	else \
		pytest -m integration -v; \
	fi

test-e2e: ## Run end-to-end tests only
	@echo "🚀 Running end-to-end tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -m e2e -v; \
	else \
		pytest -m e2e -v; \
	fi

test-cov: ## Run tests with detailed coverage report
	@echo "📊 Running tests with coverage..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest --cov=code_explainer --cov-report=html --cov-report=term-missing; \
	else \
		pytest --cov=code_explainer --cov-report=html --cov-report=term-missing; \
	fi
	@echo "📈 Coverage report generated in htmlcov/"

# Benchmarking targets
benchmark: ## Run performance benchmarks
	@echo "⚡ Running performance benchmarks..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python benchmarks/benchmark_inference.py; \
	else \
		python benchmarks/benchmark_inference.py; \
	fi

benchmark-regression: ## Run performance regression tests
	@echo "📈 Running performance regression tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest tests/test_performance_regression.py --benchmark-only -v; \
	else \
		pytest tests/test_performance_regression.py --benchmark-only -v; \
	fi

benchmark-baseline: ## Establish new performance baseline
	@echo "📊 Establishing new performance baseline..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python benchmarks/benchmark_inference.py --baseline; \
	else \
		python benchmarks/benchmark_inference.py --baseline; \
	fi

benchmark-compare: ## Compare current performance with baseline
	@echo "🔍 Comparing performance with baseline..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python benchmarks/benchmark_inference.py --compare; \
	else \
		python benchmarks/benchmark_inference.py --compare; \
	fi

benchmark-profile: ## Run profiling benchmarks with memory tracking
	@echo "🔬 Running profiling benchmarks..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest tests/test_performance_regression.py::test_memory_usage -v --profile; \
	else \
		pytest tests/test_performance_regression.py::test_memory_usage -v --profile; \
	fi

benchmark-ci: ## Run benchmarks for CI/CD (fail on regression)
	@echo "🔄 Running CI benchmarks..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python benchmarks/benchmark_inference.py --ci --fail-on-regression; \
	else \
		python benchmarks/benchmark_inference.py --ci --fail-on-regression; \
	fi

benchmark-report: ## Generate comprehensive benchmark report
	@echo "📋 Generating benchmark report..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python benchmarks/benchmark_inference.py --report; \
	else \
		python benchmarks/benchmark_inference.py --report; \
	fi

benchmark-all: ## Run all benchmarking tasks
	@echo "🚀 Running complete benchmarking suite..."
	$(MAKE) benchmark
	$(MAKE) benchmark-regression
	$(MAKE) benchmark-compare
	$(MAKE) benchmark-report

# Security scanning targets
security-scan: ## Run comprehensive security scan
	@echo "🔒 Running comprehensive security scan..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python scripts/security_scan.py; \
	else \
		python scripts/security_scan.py; \
	fi

security-vulnerabilities: ## Check for dependency vulnerabilities
	@echo "🔍 Checking for dependency vulnerabilities..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run safety check; \
	else \
		safety check || echo "⚠️  Safety not installed. Run: pip install safety"; \
	fi

security-code: ## Run code security analysis
	@echo "🔍 Running code security analysis..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run bandit -r src/; \
	else \
		bandit -r src/ || echo "⚠️  Bandit not installed. Run: pip install bandit"; \
	fi

security-semgrep: ## Run Semgrep security analysis
	@echo "🔍 Running Semgrep security analysis..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run semgrep --config auto src/; \
	else \
		semgrep --config auto src/ || echo "⚠️  Semgrep not installed. Run: pip install semgrep"; \
	fi

security-all: ## Run all security checks
	@echo "🚨 Running complete security audit..."
	$(MAKE) security-vulnerabilities
	$(MAKE) security-code
	$(MAKE) security-semgrep
	$(MAKE) security-scan

# Cache management targets
cache-stats: ## Show cache statistics and performance metrics
	@echo "📊 Getting cache statistics..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); import json; print(json.dumps(cm.get_cache_stats(), indent=2))"; \
	else \
		python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); import json; print(json.dumps(cm.get_cache_stats(), indent=2))" || echo "⚠️  Advanced caching not configured"; \
	fi

cache-clear: ## Clear all cache entries
	@echo "🗑️  Clearing all cache entries..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); cm.clear_all_caches(); print('All caches cleared')"; \
	else \
		python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); cm.clear_all_caches(); print('All caches cleared')" || echo "⚠️  Advanced caching not configured"; \
	fi

cache-invalidate: ## Invalidate cache entries by pattern (usage: make cache-invalidate PATTERN="*old*")
	@echo "🚫 Invalidating cache entries matching: $(PATTERN)"
	@if [ -z "$(PATTERN)" ]; then \
		echo "❌ Please specify PATTERN variable. Usage: make cache-invalidate PATTERN='*old*'"; \
		exit 1; \
	fi
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); count = cm.advanced_cache.invalidate_by_pattern('$(PATTERN)'); print(f'Invalidated {count} entries')"; \
	else \
		python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); count = cm.advanced_cache.invalidate_by_pattern('$(PATTERN)'); print(f'Invalidated {count} entries')" || echo "⚠️  Advanced caching not configured"; \
	fi

cache-warmup: ## Warm up cache with frequently used keys (usage: make cache-warmup KEYS="key1,key2,key3")
	@echo "🔥 Warming up cache with keys: $(KEYS)"
	@if [ -z "$(KEYS)" ]; then \
		echo "❌ Please specify KEYS variable. Usage: make cache-warmup KEYS='key1,key2,key3'"; \
		exit 1; \
	fi
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); keys = '$(KEYS)'.split(','); cm.advanced_cache.warmup(keys); print(f'Warming up {len(keys)} cache keys')"; \
	else \
		python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); keys = '$(KEYS)'.split(','); cm.advanced_cache.warmup(keys); print(f'Warming up {len(keys)} cache keys')" || echo "⚠️  Advanced caching not configured"; \
	fi

cache-backup: ## Create cache backup (usage: make cache-backup PATH="/path/to/backup")
	@echo "💾 Creating cache backup at: $(PATH)"
	@if [ -z "$(PATH)" ]; then \
		echo "❌ Please specify PATH variable. Usage: make cache-backup PATH='/tmp/cache-backup'"; \
		exit 1; \
	fi
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); cm.advanced_cache.backup('$(PATH)'); print('Cache backup created')"; \
	else \
		python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); cm.advanced_cache.backup('$(PATH)'); print('Cache backup created')" || echo "⚠️  Advanced caching not configured"; \
	fi

cache-restore: ## Restore cache from backup (usage: make cache-restore PATH="/path/to/backup")
	@echo "📂 Restoring cache from: $(PATH)"
	@if [ -z "$(PATH)" ]; then \
		echo "❌ Please specify PATH variable. Usage: make cache-restore PATH='/tmp/cache-backup'"; \
		exit 1; \
	fi
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); cm.advanced_cache.restore('$(PATH)'); print('Cache restored')"; \
	else \
		python -c "from src.code_explainer.advanced_cache import CacheManager; cm = CacheManager(); cm.advanced_cache.restore('$(PATH)'); print('Cache restored')" || echo "⚠️  Advanced caching not configured"; \
	fi

cache-all: ## Run all cache management operations
	@echo "🔄 Running all cache management operations..."
	$(MAKE) cache-stats
	@echo "--- Cache operations completed ---"

cache-test: ## Run advanced caching tests
	@echo "🧪 Running advanced caching tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python scripts/test_advanced_cache.py; \
	else \
		python scripts/test_advanced_cache.py; \
	fi

# Quality assurance targets
check: ## Run all quality checks (lint, type, format)
	@echo "🔍 Running quality checks..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type

quality: ## Run comprehensive quality checks
	@echo "✨ Running comprehensive quality checks..."
	$(MAKE) check
	$(MAKE) precommit
	$(MAKE) test-unit
	@echo "✅ All quality checks passed!"

setup: ## Complete development environment setup
	@echo "🚀 Setting up complete development environment..."
	$(MAKE) install-dev
	$(MAKE) validate-env
	@echo "💡 Run 'make quality' to verify everything works"
	@echo "📚 Run 'make docs-serve' to view documentation"

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

docs-build: ## Build documentation
	mkdocs build

# Docker targets
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t code-explainer .

docker-build-dev: ## Build development Docker image
	@echo "🐳 Building development Docker image..."
	docker build --target development -t code-explainer:dev .

docker-build-prod: ## Build production Docker image
	@echo "🐳 Building production Docker image..."
	docker build --target production -t code-explainer:prod .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 8000:8000 -v $(PWD):/app code-explainer

docker-dev: ## Run Docker container in development mode
	@echo "🐳 Running Docker container in development mode..."
	docker run -p 8000:8000 -p 8001:8001 -v $(PWD):/app -e DEV=true code-explainer:dev

docker-compose-up: ## Start all services with docker-compose
	@echo "🐳 Starting services with docker-compose..."
	docker-compose up -d

docker-compose-dev: ## Start development services with docker-compose
	@echo "🐳 Starting development services..."
	docker-compose --profile docs up -d api web docs

docker-compose-test: ## Run tests in Docker container
	@echo "🧪 Running tests in Docker..."
	docker-compose --profile testing run --rm test

docker-compose-down: ## Stop all docker-compose services
	@echo "🐳 Stopping services..."
	docker-compose down

docker-clean: ## Clean Docker artifacts
	@echo "🧹 Cleaning Docker artifacts..."
	docker system prune -f
	docker image prune -f

# Release management
version: ## Show current version
	@if command -v poetry >/dev/null 2>&1; then \
		poetry version; \
	else \
		python -c "import toml; print(toml.load('pyproject.toml')['tool']['poetry']['version'])" 2>/dev/null || \
		python setup.py --version 2>/dev/null || \
		echo "Unable to determine version"; \
	fi

bump-major: ## Bump major version
	@if command -v poetry >/dev/null 2>&1; then \
		poetry version major; \
	else \
		echo "❌ Poetry required for version bumping"; \
	fi

bump-minor: ## Bump minor version
	@if command -v poetry >/dev/null 2>&1; then \
		poetry version minor; \
	else \
		echo "❌ Poetry required for version bumping"; \
	fi

bump-patch: ## Bump patch version
	@if command -v poetry >/dev/null 2>&1; then \
		poetry version patch; \
	else \
		echo "❌ Poetry required for version bumping"; \
	fi

release: ## Create and publish a new release
	@echo "🚀 Creating release..."
	$(MAKE) quality
	$(MAKE) test-all
	@echo "📦 Building distribution..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry build; \
		echo "📤 Publishing to PyPI..."; \
		poetry publish; \
	else \
		python setup.py sdist bdist_wheel; \
		echo "📤 Upload to PyPI manually: twine upload dist/*"; \
	fi
	@echo "✅ Release complete!"

# API Development Commands
.PHONY: api-dev api-prod api-test api-reload api-metrics

api-dev: ## Start API server in development mode
	@echo "🚀 Starting API server (development)..."
	uvicorn code_explainer.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Start API server in production mode
	@echo "🚀 Starting API server (production)..."
	uvicorn code_explainer.api.main:app --host 0.0.0.0 --port 8000 --workers 4

api-test: ## Test API endpoints
	@echo "🧪 Testing API endpoints..."
	@curl -f http://localhost:8000/health || (echo "❌ API not responding" && exit 1)
	@echo "\n✅ Health check passed"
	@curl -f http://localhost:8000/version
	@echo "\n✅ Version endpoint working"

api-reload: ## Trigger model reload via API (requires API key)
	@echo "🔄 Reloading model..."
	@curl -X POST http://localhost:8000/admin/reload \
		-H "X-API-Key: ${CODE_EXPLAINER_API_KEY}" || echo "Set CODE_EXPLAINER_API_KEY environment variable"

api-metrics: ## Show API metrics
	@echo "📊 Fetching metrics..."
	@curl -s http://localhost:8000/metrics | python -m json.tool

# Docker shortcuts
.PHONY: docker-up docker-down docker-logs docker-ps docker-shell

docker-up: ## Start all services with docker-compose
	@echo "🐳 Starting services..."
	docker-compose up -d
	@echo "✅ Services started. API: http://localhost:8000, Streamlit: http://localhost:8501"

docker-down: ## Stop all services
	@echo "🛑 Stopping services..."
	docker-compose down

docker-logs: ## Show logs from all services
	docker-compose logs -f

docker-ps: ## Show running containers
	docker-compose ps

docker-shell: ## Open shell in API container
	docker-compose exec api bash

# Monitoring shortcuts
.PHONY: monitoring-up monitoring-down prometheus-ui grafana-ui

monitoring-up: ## Start monitoring stack (Prometheus + Grafana)
	@echo "📊 Starting monitoring stack..."
	docker-compose --profile monitoring up -d
	@echo "✅ Prometheus: http://localhost:9090"
	@echo "✅ Grafana: http://localhost:3000 (admin/admin)"

monitoring-down: ## Stop monitoring stack
	docker-compose --profile monitoring down

prometheus-ui: ## Open Prometheus UI in browser
	@echo "📊 Opening Prometheus..."
	@open http://localhost:9090 2>/dev/null || xdg-open http://localhost:9090 2>/dev/null || echo "Open http://localhost:9090"

grafana-ui: ## Open Grafana UI in browser
	@echo "📊 Opening Grafana..."
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Open http://localhost:3000"

# Quick start shortcuts
.PHONY: quickstart demo

quickstart: install-dev ## Quick start: install and run demo
	@echo "🎬 Running quick start..."
	@$(MAKE) api-dev &
	@sleep 5
	@$(MAKE) api-test
	@echo "✅ Quick start complete! API running at http://localhost:8000"

demo: ## Run interactive demo
	@echo "🎬 Starting demo..."
	python -m code_explainer.cli explain "def hello(): print('world')"
