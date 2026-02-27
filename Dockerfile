# Multi-stage Docker build for Code Explainer
# Stage 1: Base dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r codeexplainer && useradd -r -g codeexplainer codeexplainer

# Stage 2: Development environment
FROM base as development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set development environment variables
ENV DEV=true \
    PYTHONPATH=/app/src

# Create workspace directory
RUN mkdir -p /app && chown codeexplainer:codeexplainer /app
WORKDIR /app

# Copy dependency files first for better caching
COPY --chown=codeexplainer:codeexplainer pyproject.toml poetry.lock requirements*.txt ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -e .[dev,web,rag,metrics]

# Copy source code
COPY --chown=codeexplainer:codeexplainer . .

# Switch to non-root user
USER codeexplainer

# Health check for development
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports for API and docs
EXPOSE 8000 8001 7860

# Default command for development with hot reload
CMD ["uvicorn", "code_explainer.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src"]

# Stage 3: Production environment
FROM base as production

# Set production environment variables
ENV PYTHONPATH=/app/src \
    CODE_EXPLAINER_MODEL_PATH=/app/models \
    CODE_EXPLAINER_CONFIG_PATH=/app/configs/default.yaml

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/configs /app/.cache && \
    chown -R codeexplainer:codeexplainer /app

WORKDIR /app

# Copy dependency files
COPY --chown=codeexplainer:codeexplainer pyproject.toml poetry.lock requirements*.txt ./

# Install production dependencies only
RUN pip install --upgrade pip \
    && pip install -e .[web] \
    && pip install --force-reinstall --no-deps uvicorn[standard] \
    && find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy source code
COPY --chown=codeexplainer:codeexplainer src/ ./src/
COPY --chown=codeexplainer:codeexplainer configs/ ./configs/

# Switch to non-root user with security hardening
USER codeexplainer

# Security: drop all Linux capabilities, prevent privilege escalation
# (requires Docker --security-opt no-new-privileges at runtime)

# Health check for production (use curl instead of python for lighter footprint)
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command for production with access logging
CMD ["uvicorn", "code_explainer.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--access-log", "--limit-concurrency", "100"]

# Stage 4: CI/Testing environment
FROM development as testing

# Install additional testing dependencies
RUN pip install pytest pytest-cov pytest-xdist

# Set testing environment variables
ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTEST_PLUGINS=pytest_cov.plugin

# Default command for testing
CMD ["pytest", "--cov=code_explainer", "--cov-report=xml", "--cov-report=term-missing"]

# Default to development stage if no target specified
FROM development
