"""Data models for retrieval system.

Optimized with:
- __future__ annotations for deferred type evaluation
- __slots__ for memory efficiency  
- frozen=True for immutable config
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval behavior (immutable for hashability)."""
    batch_size: int = 32
    hybrid_alpha: float = 0.5


# Pre-create default method usage dict to avoid repeated lambda calls
_DEFAULT_METHOD_USAGE: Dict[str, int] = {"faiss": 0, "bm25": 0, "hybrid": 0}


@dataclass
class RetrievalStats:
    """Statistics for retrieval operations."""
    total_queries: int = 0
    method_usage: Dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_METHOD_USAGE))
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
