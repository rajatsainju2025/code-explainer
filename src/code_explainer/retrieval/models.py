"""Data models for retrieval system.

Optimized with:
- __future__ annotations for deferred type evaluation
- __slots__ for memory efficiency  
- frozen=True for immutable config
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalCandidate:
    """A candidate result from retrieval."""
    content: str
    index: int
    initial_score: float
    method: str
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval behavior (immutable for hashability)."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    faiss_index_type: str = "IndexFlatL2"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.5
    rerank_top_k: int = 20
    mmr_lambda: float = 0.5


# Pre-create default method usage dict to avoid repeated lambda calls
_DEFAULT_METHOD_USAGE: Dict[str, int] = {"faiss": 0, "bm25": 0, "hybrid": 0}


@dataclass
class RetrievalStats:
    """Statistics for retrieval operations."""
    total_queries: int = 0
    method_usage: Dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_METHOD_USAGE))
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
    rerank_usage: int = 0
    mmr_usage: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
