"""Legacy setuptools configuration for pip compatibility."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements from pyproject.toml is preferred, but provide fallback
requirements = [
    "torch>=2.0.0,<3.0.0",
    "transformers>=4.30.0,<5.0.0", 
    "datasets>=2.12.0,<3.0.0",
    "accelerate>=0.20.0,<1.0.0",
    "tokenizers>=0.13.0,<1.0.0",
    "pandas>=2.0.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
    "PyYAML>=6.0,<7.0.0",
    "psutil>=5.9.0,<6.0.0",
    "click>=8.1.0,<9.0.0",
    "tqdm>=4.65.0,<5.0.0",
    "rich>=13.0.0,<14.0.0",
    "hydra-core>=1.3.0,<2.0.0",
    "omegaconf>=2.3.0,<3.0.0",
    "pydantic>=2.0.0,<3.0.0",
]

setup(
    name="code-explainer",
    version="0.3.0",
    author="Rajat Sainju",
    author_email="your.email@example.com",
    description="An efficient LLM-powered tool for explaining code snippets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajatsainju2025/code-explainer",
    project_urls={
        "Documentation": "https://github.com/rajatsainju2025/code-explainer#readme",
        "Source": "https://github.com/rajatsainju2025/code-explainer",
        "Issues": "https://github.com/rajatsainju2025/code-explainer/issues",
        "Discussions": "https://github.com/rajatsainju2025/code-explainer/discussions",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Documentation",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "web": [
            "gradio>=3.35.0,<4.0.0",
            "streamlit>=1.24.0,<2.0.0",
            "fastapi>=0.100.0,<1.0.0",
            "uvicorn>=0.22.0,<1.0.0",
            "prometheus-client>=0.17.0,<1.0.0",
            "requests>=2.31.0,<3.0.0",
        ],
        "rag": [
            "sentence-transformers>=2.2.0,<3.0.0",
            "faiss-cpu>=1.7.0,<2.0.0",
            "rank-bm25>=0.2.2,<1.0.0",
            "scikit-learn>=1.3.0,<2.0.0",
        ],
        "metrics": [
            "sacrebleu>=2.4.0,<3.0.0",
            "rouge-score>=0.1.2,<1.0.0",
            "bert-score>=0.3.13,<1.0.0",
            "codebleu>=0.6.0,<1.0.0",
        ],
        "monitoring": [
            "wandb>=0.15.0,<1.0.0",
            "tensorboard>=2.13.0,<3.0.0",
        ],
        "all": [
            # Web dependencies
            "gradio>=3.35.0,<4.0.0",
            "streamlit>=1.24.0,<2.0.0",
            "fastapi>=0.100.0,<1.0.0",
            "uvicorn>=0.22.0,<1.0.0",
            "prometheus-client>=0.17.0,<1.0.0",
            "requests>=2.31.0,<3.0.0",
            # RAG dependencies
            "sentence-transformers>=2.2.0,<3.0.0",
            "faiss-cpu>=1.7.0,<2.0.0",
            "rank-bm25>=0.2.2,<1.0.0",
            "scikit-learn>=1.3.0,<2.0.0",
            # Metrics dependencies
            "sacrebleu>=2.4.0,<3.0.0",
            "rouge-score>=0.1.2,<1.0.0",
            "bert-score>=0.3.13,<1.0.0",
            "codebleu>=0.6.0,<1.0.0",
            # Monitoring dependencies
            "wandb>=0.15.0,<1.0.0",
            "tensorboard>=2.13.0,<3.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "code-explainer=code_explainer.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "code_explainer": ["configs/*.yaml"],
    },
)