#!/usr/bin/env python3
"""
Smart setup script for Code Explainer.

This script automatically detects your environment and installs
dependencies using the best available method (Poetry or pip).
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Union


def run_command(cmd: List[str], check: bool = True, capture_output: bool = False) -> Union[subprocess.CompletedProcess, subprocess.CalledProcessError]:
    """Run a command and handle errors gracefully."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
            cwd=Path(__file__).parent
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"Error: {e}")
        if not check:
            return e
        sys.exit(1)


def has_command(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def detect_python_version() -> tuple:
    """Detect Python version."""
    return sys.version_info[:2]


def check_python_compatibility():
    """Check if Python version is compatible."""
    version = detect_python_version()
    if version < (3, 9):
        print(f"❌ Python {version[0]}.{version[1]} detected.")
        print("Code Explainer requires Python 3.9 or higher.")
        print("Please upgrade your Python version.")
        sys.exit(1)
    print(f"✅ Python {version[0]}.{version[1]} detected - compatible!")


def detect_compute_devices():
    """Detect available compute devices and provide setup hints."""
    try:
        import torch
        print("🖥️  Device detection:")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ CUDA GPU: {gpu_name} ({memory_gb:.1f} GB)")

            if memory_gb < 6:
                print("  💡 Tip: Consider using 8-bit quantization for this GPU")
                print("       Set CODE_EXPLAINER_PRECISION=8bit")

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✅ Apple Silicon (MPS) detected")
            print("  💡 Tip: MPS works best with fp16 precision")

        else:
            print("  ✅ CPU-only mode")
            print("  💡 Tip: CPU inference is slower but works on all devices")

    except ImportError:
        print("  ⚠️  PyTorch not yet installed - device detection skipped")


def install_with_poetry(profile: str = "full"):
    """Install using Poetry."""
    print("🎯 Installing with Poetry...")

    if not has_command("poetry"):
        print("❌ Poetry not found.")
        print("Install Poetry with: curl -sSL https://install.python-poetry.org | python3 -")
        return False

    # Configure poetry to create venv in project
    run_command(["poetry", "config", "virtualenvs.in-project", "true"], check=False)

    # Install based on profile
    if profile == "minimal":
        run_command(["poetry", "install", "--only", "main"])
    elif profile == "dev":
        run_command(["poetry", "install", "--all-extras", "--with", "dev"])
        # Set up pre-commit if available
        if has_command("pre-commit"):
            run_command(["poetry", "run", "pre-commit", "install"], check=False)
    else:  # full
        run_command(["poetry", "install", "--all-extras"])

    print("✅ Poetry installation complete!")
    return True


def install_with_pip(profile: str = "full"):
    """Install using pip."""
    print("📦 Installing with pip...")

    # Upgrade pip
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install based on profile
    if profile == "minimal":
        if Path("requirements-core.txt").exists():
            run_command([sys.executable, "-m", "pip", "install", "-r", "requirements-core.txt"])
        else:
            run_command([sys.executable, "-m", "pip", "install", "-e", "."])
    elif profile == "dev":
        run_command([sys.executable, "-m", "pip", "install", "-e", ".[all]"])
        # Install dev requirements if available
        for req_file in ["requirements-dev.txt", "requirements.txt"]:
            if Path(req_file).exists():
                run_command([sys.executable, "-m", "pip", "install", "-r", req_file], check=False)
                break
    else:  # full
        run_command([sys.executable, "-m", "pip", "install", "-e", ".[all]"])

    print("✅ Pip installation complete!")
    return True


def setup_environment_variables():
    """Provide guidance on environment variables."""
    print("\n🔧 Environment Configuration:")
    print("You can customize Code Explainer with these environment variables:")
    print("")
    print("  CODE_EXPLAINER_DEVICE=auto|cuda|mps|cpu")
    print("    Force specific device (default: auto-detect)")
    print("")
    print("  CODE_EXPLAINER_PRECISION=auto|fp32|fp16|bf16|8bit")
    print("    Control model precision (default: auto-optimize)")
    print("")
    print("  CODE_EXPLAINER_FALLBACK_ENABLED=true|false")
    print("    Enable CPU fallback on GPU errors (default: true)")
    print("")
    print("Add these to your ~/.bashrc, ~/.zshrc, or .env file")


def validate_installation():
    """Validate that the installation worked."""
    print("\n🔍 Validating installation...")

    try:
        # Test basic imports
        import torch
        import transformers
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  ✅ Transformers {transformers.__version__}")

        # Test our package
        from src.code_explainer.device_manager import device_manager
        device_info = device_manager.get_device_info()
        optimal_device = device_manager.get_optimal_device()
        print(f"  ✅ Code Explainer modules loaded")
        print(f"  ✅ Optimal device: {optimal_device.device_type}")

        print("🎉 Installation validation successful!")
        return True

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("Installation may be incomplete. Try running the setup again.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Smart setup for Code Explainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                    # Auto-detect, install full version
  python setup.py --method poetry    # Force Poetry installation
  python setup.py --profile minimal  # Install minimal dependencies only
  python setup.py --profile dev      # Install development environment
        """
    )
    parser.add_argument(
        "--method",
        choices=["auto", "poetry", "pip"],
        default="auto",
        help="Installation method (default: auto-detect)"
    )
    parser.add_argument(
        "--profile",
        choices=["minimal", "full", "dev"],
        default="full",
        help="Installation profile (default: full)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip installation validation"
    )

    args = parser.parse_args()

    print("🚀 Code Explainer Setup")
    print("=" * 40)

    # Check Python compatibility
    check_python_compatibility()

    # Detect devices for setup hints
    detect_compute_devices()

    # Choose installation method
    success = False
    if args.method == "auto":
        if has_command("poetry"):
            print("\n📦 Auto-detected: Using Poetry")
            success = install_with_poetry(args.profile)
        else:
            print("\n📦 Auto-detected: Using pip")
            success = install_with_pip(args.profile)
    elif args.method == "poetry":
        success = install_with_poetry(args.profile)
    else:  # pip
        success = install_with_pip(args.profile)

    if not success:
        print("❌ Installation failed!")
        sys.exit(1)

    # Validate installation
    if not args.skip_validation:
        if not validate_installation():
            sys.exit(1)

    # Provide environment setup guidance
    setup_environment_variables()

    print(f"\n✅ Setup complete! Profile: {args.profile}")
    print("\n🎯 Next steps:")
    print("  1. Try: python -m src.code_explainer.cli --help")
    print("  2. Run tests: make test")
    print("  3. Check device info: make device-info")
    print("  4. Read docs: make docs-serve")


if __name__ == "__main__":
    main()
