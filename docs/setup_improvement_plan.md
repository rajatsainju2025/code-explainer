# Setup and Dependency Management Improvement Plan

## Current Issues Identified

### 1. Mixed Dependency Management
- **Poetry configuration**: pyproject.toml with organized dependency groups
- **Legacy pip requirements**: Multiple requirements-*.txt files
- **Makefile inconsistency**: Uses `pip install -e .[dev]` but there are no extras in setup.py
- **Inconsistent workflows**: Poetry lock file exists but Makefile doesn't use poetry

### 2. Environment Setup Problems
- No clear documentation for different environment types (minimal/full/development)
- Missing environment variable support documentation
- No requirements.txt generation from Poetry for users who prefer pip
- setup.py exists but appears unused

### 3. Makefile Issues
- `pip install -e .[dev]` fails because setup.py doesn't define extras
- No Poetry commands in Makefile
- Missing Docker/container setup commands
- No environment validation checks

## Proposed Solutions

### 1. Standardize on Poetry (Recommended)
**Benefits:**
- Better dependency resolution
- Clear separation of dev/optional dependencies
- Deterministic builds with poetry.lock
- Modern Python packaging standards

**Changes:**
1. Update Makefile to use Poetry commands
2. Add convenience scripts for pip-only users
3. Generate requirements.txt files from Poetry
4. Remove redundant requirements files
5. Fix setup.py or remove it entirely

### 2. Environment Variable Integration
Add support for:
- `POETRY_VENV_IN_PROJECT=true` - Keep venv in project directory
- `POETRY_CACHE_DIR` - Configure cache location
- `POETRY_NO_DEV=true` - Skip dev dependencies for production

### 3. Installation Profiles
Create easy installation profiles:
- **Minimal**: Core functionality only
- **Full**: All optional features (web, rag, metrics)
- **Development**: All deps + dev tools
- **Production**: Optimized for deployment

### 4. Enhanced Makefile
Update with:
- Poetry-first commands with pip fallbacks
- Environment validation
- Docker integration
- Cross-platform support
- Device-specific setup hints

## Implementation Plan

1. **Update pyproject.toml**: Clean up dependencies, fix extras
2. **Create setup script**: Smart installer that detects environment
3. **Update Makefile**: Poetry-first with fallbacks
4. **Generate requirements**: Auto-generate from Poetry
5. **Add environment validation**: Check system compatibility
6. **Update documentation**: Clear setup instructions
7. **Add Docker support**: Multi-stage builds for different profiles

## Migration Strategy

1. Keep existing requirements.txt as fallback during transition
2. Add deprecation warnings for old setup methods
3. Provide migration guide for existing users
4. Test on multiple platforms (macOS, Linux, Windows)
5. Update CI/CD to use new setup process