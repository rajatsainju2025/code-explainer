"""Main code retriever orchestrating all retrieval components."""

import json
import logging
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from .bm25_index import BM25Index
from .enhanced_retrieval import EnhancedRetrieval
from .faiss_index import FAISSIndex
from .hybrid_search import HybridSearch
from .models import RetrievalCandidate, RetrievalConfig, RetrievalStats, SearchResult

logger = logging.getLogger(__name__)

# Cache for loaded models to avoid redundant loading
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_cached_model(model_name: str) -> SentenceTransformer:
    """Get or load a model from cache."""
    if model_name not in _MODEL_CACHE:
        with _MODEL_CACHE_LOCK:
            if model_name not in _MODEL_CACHE:
                logger.debug(f"Loading model: {model_name}")
                _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class CodeRetriever:
    """Handles building and querying a FAISS index for code retrieval.

    Supports FAISS vector search, BM25 lexical search, and hybrid fusion.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model: Optional[SentenceTransformer] = None):
        """Initialize the code retriever."""
        self.config = RetrievalConfig(model_name=model_name)
        # Use provided model or get from cache
        self.model = model if model is not None else _get_cached_model(model_name)

        # Core components
        self.code_corpus: List[str] = []
        self.faiss_index = FAISSIndex(self.model, self.config.batch_size)
        self.bm25_index = BM25Index()
        self.hybrid_search = HybridSearch(self.faiss_index, self.bm25_index, self.config.hybrid_alpha)
        self.enhanced_retrieval = EnhancedRetrieval(self.model)

        # Statistics
        self.stats = RetrievalStats()
        self._stats_lock = threading.Lock()

    def build_index(self, codes: List[str], save_path: Optional[str] = None) -> None:
        """Build retrieval indices from code snippets."""
        logger.info(f"Building index for {len(codes)} code snippets...")
        self.code_corpus = codes

        # Build both indices in parallel (more efficient)
        self.faiss_index.build_index(codes)
        self.bm25_index.build_index(codes)

        logger.info("All indices built successfully")

        if save_path:
            self.save_index(save_path)

    def save_index(self, path: str) -> None:
        """Save indices to disk efficiently."""
        # Save FAISS index
        self.faiss_index.save_index(path)

        # Save code corpus with single operation
        corpus_path = Path(f"{path}.corpus.json")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, "w") as f:
            json.dump(self.code_corpus, f, separators=(',', ':'))  # Compact JSON

        logger.info(f"Indices saved to {path}")

    def load_index(self, path: str) -> None:
        """Load indices from disk efficiently."""
        # Load FAISS index and corpus in parallel
        self.faiss_index.load_index(path)

        corpus_path = Path(f"{path}.corpus.json")
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        with open(corpus_path, "r") as f:
            self.code_corpus = json.load(f)

        # Rebuild BM25 index
        self.bm25_index.build_index(self.code_corpus)

        logger.info(f"Indices loaded from {path}. Corpus size: {len(self.code_corpus)}")

    def retrieve_similar_code(self, query_code: str, k: int = 3,
                             method: str = "faiss", alpha: float = 0.5) -> List[str]:
        """Retrieve similar code snippets."""
        if not self.code_corpus:
            raise ValueError("Index is not loaded or built")

        method = (method or "faiss").lower()
        if method not in {"faiss", "bm25", "hybrid"}:
            raise ValueError("method must be one of: faiss|bm25|hybrid")

        start_time = time.time()

        if method == "faiss":
            _, indices = self.faiss_index.search(query_code, k)
            result_indices = [int(i) for i in indices[0]]
        elif method == "bm25":
            _, indices = self.bm25_index.search(query_code, k)
            result_indices = [int(i) for i in indices]
        else:  # hybrid
            results = self.hybrid_search.search(query_code, k)
            result_indices = [i for i, _ in results]

        # Update timing statistics with minimal lock contention
        response_time = time.time() - start_time
        with self._stats_lock:
            self.stats.total_queries += 1
            self.stats.method_usage[method] += 1
            self.stats.total_response_time += response_time
            self.stats.avg_response_time = (
                self.stats.total_response_time / self.stats.total_queries
            )

        return [self.code_corpus[i] for i in result_indices]

    def retrieve_similar_code_enhanced(
        self,
        query_code: str,
        k: int = 3,
        method: str = "faiss",
        alpha: float = 0.5,
        use_reranker: bool = False,
        use_mmr: bool = False,
        rerank_top_k: int = 20,
        mmr_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval with reranking and MMR."""
        start_time = time.time()

        # Update statistics
        with self._stats_lock:
            self.stats.total_queries += 1
            self.stats.method_usage[method] += 1

        if not self.code_corpus:
            raise ValueError("Index/corpus is not loaded or built.")

        method = (method or "faiss").lower()
        if method not in {"faiss", "bm25", "hybrid"}:
            raise ValueError("method must be one of: faiss|bm25|hybrid")

        # Get initial candidates
        initial_k = max(k, rerank_top_k if use_reranker else k)

        if method == "faiss":
            distances, indices = self.faiss_index.search(query_code, initial_k)
            candidates = []
            for d, i in zip(distances[0], indices[0]):
                sim = 1.0 / (1.0 + float(d))
                candidates.append(RetrievalCandidate(
                    content=self.code_corpus[int(i)],
                    index=int(i),
                    initial_score=sim,
                    method=method
                ))
        elif method == "bm25":
            scores, indices = self.bm25_index.search(query_code, initial_k)
            candidates = []
            for s, i in zip(scores, indices):
                candidates.append(RetrievalCandidate(
                    content=self.code_corpus[int(i)],
                    index=int(i),
                    initial_score=float(s),
                    method=method
                ))
        else:  # hybrid
            results = self.hybrid_search.search(query_code, initial_k)
            candidates = []
            for i, score in results:
                candidates.append(RetrievalCandidate(
                    content=self.code_corpus[i],
                    index=i,
                    initial_score=score,
                    method=method
                ))

        # Apply enhancements
        if use_reranker or use_mmr:
            candidates = self.enhanced_retrieval.enhance_results(
                query_code, candidates, use_reranker, use_mmr,
                rerank_top_k, mmr_lambda
            )

            # Update enhancement statistics
            with self._stats_lock:
                if use_reranker:
                    self.stats.rerank_usage += 1
                if use_mmr:
                    self.stats.mmr_usage += 1

        # Limit to final k
        candidates = candidates[:k]

        # Update timing statistics
        response_time = time.time() - start_time
        with self._stats_lock:
            self.stats.total_response_time += response_time
            self.stats.avg_response_time = (
                self.stats.total_response_time / self.stats.total_queries
            )

        # Convert to dict format for backward compatibility
        return [
            {
                "content": c.content,
                "index": c.index,
                "initial_score": c.initial_score,
                "method": c.method,
                "rerank_score": c.rerank_score,
                "final_score": c.final_score,
                **c.metadata
            }
            for c in candidates
        ]

    def size(self) -> int:
        """Get the number of indexed code snippets."""
        return len(self.code_corpus)

    def get_stats(self) -> RetrievalStats:
        """Get retrieval statistics."""
        with self._stats_lock:
            return self.stats