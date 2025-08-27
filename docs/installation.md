# Installation Guide

This guide covers different ways to install and set up Code Explainer.

## Prerequisites

- Python 3.8 or higher
- Git (for source installation)
- Docker (optional, for containerized deployment)

## Quick Installation

### Option 1: pip install (Recommended)

```bash
pip install code-explainer
```

### Option 2: From Source

```bash
git clone https://github.com/rajatsainju2025/code-explainer.git
cd code-explainer
make install
```

### Option 3: Docker

```bash
docker run -p 8000:8000 rajatsainju/code-explainer:latest
```

## Development Installation

For development and contributing:

```bash
# Clone the repository
git clone https://github.com/rajatsainju2025/code-explainer.git
cd code-explainer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
make dev-install

# Install pre-commit hooks
make hooks
```

## Verification

Test your installation:

```bash
# Test CLI
code-explainer --version

# Test Python import
python -c "from code_explainer import CodeExplainer; print('Success!')"

# Test API server
make serve
# Visit http://localhost:8000/docs
```

## GPU Support

For GPU acceleration (optional):

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Common Issues

1. **Python version**: Ensure Python 3.8+
2. **Virtual environment**: Always use a virtual environment
3. **Dependencies**: Run `make install` to ensure all dependencies
4. **Permissions**: Use `pip install --user` if permission errors

### Platform-specific Notes

#### macOS
```bash
# Install Xcode command line tools if needed
xcode-select --install
```

#### Windows
```bash
# Use PowerShell or Command Prompt
# Ensure Microsoft Visual C++ 14.0 is installed
```

#### Linux
```bash
# Install build essentials
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

## Next Steps

After installation, check out:

- [Quick Start Guide](quickstart.md) for basic usage
- [Configuration](configuration.md) for setup options
- [API Reference](api/rest.md) for detailed documentation
