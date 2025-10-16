"""
Enhanced code retrieval with FAISS, BM25, hybrid search, reranking, and MMR.

NOTE: This module has been refactored into a modular structure.
All functionality is now available through the retrieval package.
"""

# Import everything from the new modular structure for backward compatibility
from .retrieval import (
    BM25Index,
    CodeRetriever,
    EnhancedRetrieval,
    FAISSIndex,
    HybridSearch,
    RetrievalCandidate,
    RetrievalConfig,
    RetrievalStats,
    SearchResult,
    TextTokenizer,
    tokenizer,
)

__all__ = [
    "CodeRetriever",
    "BM25Index", 
    "FAISSIndex",
    "HybridSearch",
    "EnhancedRetrieval",
    "RetrievalCandidate",
    "RetrievalConfig",
    "RetrievalStats",
    "SearchResult",
    "TextTokenizer",
    "tokenizer",
]
