"""Data models for retrieval system."""

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


@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    faiss_index_type: str = "IndexFlatL2"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.5
    rerank_top_k: int = 20
    mmr_lambda: float = 0.5


@dataclass
class RetrievalStats:
    """Statistics for retrieval operations."""
    total_queries: int = 0
    method_usage: Dict[str, int] = field(default_factory=lambda: {"faiss": 0, "bm25": 0, "hybrid": 0})
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
    rerank_usage: int = 0
    mmr_usage: int = 0


@dataclass
class SearchResult:
    """Result of a search operation."""
    query: str
    method: str
    candidates: List[RetrievalCandidate]
    response_time: float
    total_candidates: int