"""Type hints and annotations for Code Explainer.

Provides:
- Type aliases for common patterns
- Protocol definitions for duck typing
- Literal types for constrained values
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, Union

try:
    from typing import TypeAlias
except ImportError:
    # Python < 3.10
    TypeAlias = type  # type: ignore[misc,assignment]

# Strategy type aliases
StrategyName: TypeAlias = Literal[
    "basic",
    "detailed",
    "beginner",
    "advanced",
    "security",
    "performance",
    "complexity",
    "docstring"
]

# Model type aliases
ModelName: TypeAlias = Literal[
    "codet5-small",
    "codet5-base",
    "codebert-base",
    "codellama-instruct",
    "starcoder2-instruct",
    "starcoderbase-1b"
]

# Device type
Device: TypeAlias = Literal["cpu", "cuda", "mps"]

# Output format
OutputFormat: TypeAlias = Literal["text", "json", "markdown"]

# Retrieval mode
RetrievalMode: TypeAlias = Literal["disabled", "simple", "enhanced", "hybrid"]


class ExplanationProtocol(Protocol):
    """Protocol for explanation objects."""
    
    code: str
    explanation: str
    strategy: str
    model: Union[str, None]
    metadata: Union[dict, None]


class ModelProtocol(Protocol):
    """Protocol for model objects."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        ...
    
    def to(self, device: str) -> "ModelProtocol":
        """Move model to device."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retriever objects."""
    
    def retrieve(self, query: str, top_k: int = 5) -> Sequence[dict]:
        """Retrieve relevant documents."""
        ...
    
    def add_documents(self, documents: Sequence[str]) -> None:
        """Add documents to the retriever."""
        ...
