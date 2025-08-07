"""Setup script for code-explainer package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="code-explainer",
    version="0.1.0",
    author="Rajat Sainju",
    author_email="your.email@example.com",
    description="An efficient LLM-powered tool for explaining code snippets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajatsainju2025/code-explainer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Documentation",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0", 
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "web": [
            "gradio>=3.35.0",
            "streamlit>=1.24.0", 
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "code-explainer=code_explainer.cli:main",
            "train-code-explainer=code_explainer.cli:train",
        ],
    },
    include_package_data=True,
    package_data={
        "code_explainer": ["configs/*.yaml"],
    },
)
