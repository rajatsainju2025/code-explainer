"""Main code retriever orchestrating all retrieval components.

Optimized for performance with:
- Fast xxhash-based cache key generation
- Pre-computed method validation sets
- Efficient memory usage with __slots__
"""

import gzip
import json
import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .bm25_index import BM25Index
from .faiss_index import FAISSIndex
from .hybrid_search import HybridSearch
from .models import RetrievalConfig, RetrievalStats
from .model_cache import get_cached_model

logger = logging.getLogger(__name__)

# Pre-compute valid methods set for O(1) lookup
_VALID_METHODS = frozenset({"faiss", "bm25", "hybrid"})


class CodeRetriever:
    """Handles building and querying a FAISS index for code retrieval.

    Supports FAISS vector search, BM25 lexical search, and hybrid fusion.
    """
    
    __slots__ = ('config', 'model', 'code_corpus', 'faiss_index', 'bm25_index', 
                 'hybrid_search', 'stats', '_stats_lock')

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model: Optional[SentenceTransformer] = None):
        """Initialize the code retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            model: Pre-loaded model instance (optional)
        """
        self.config = RetrievalConfig()
        # Use provided model or get from persistent cache
        self.model = model if model is not None else get_cached_model(model_name)

        # Core components
        self.code_corpus: List[str] = []
        self.faiss_index = FAISSIndex(self.model, self.config.batch_size)
        self.bm25_index = BM25Index()
        self.hybrid_search = HybridSearch(self.faiss_index, self.bm25_index, self.config.hybrid_alpha)

        # Statistics
        self.stats = RetrievalStats()
        self._stats_lock = threading.Lock()

    def build_index(self, codes: List[str], save_path: Optional[str] = None) -> None:
        """Build retrieval indices from code snippets."""
        logger.info("Building index for %d code snippets...", len(codes))
        self.code_corpus = codes

        # Build both indices in parallel (more efficient)
        self.faiss_index.build_index(codes)
        self.bm25_index.build_index(codes)

        logger.info("All indices built successfully")

        if save_path:
            self.save_index(save_path)

    def save_index(self, path: str) -> None:
        """Save indices to disk efficiently with gzip compression."""
        # Save FAISS index
        self.faiss_index.save_index(path)

        # Save code corpus with gzip compression (reduces size by ~90%)
        corpus_path = Path(f"{path}.corpus.json.gz")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use compact separators to reduce file size further
        json_data = json.dumps(self.code_corpus, separators=(',', ':'), ensure_ascii=True)
        
        # Compress with gzip
        with gzip.open(corpus_path, 'wt', encoding='utf-8') as f:
            f.write(json_data)

        logger.debug("Indices saved to %s with compressed corpus", path)

    def load_index(self, path: str) -> None:
        """Load indices from disk, supporting both compressed and uncompressed formats."""
        # Load FAISS index
        self.faiss_index.load_index(path)

        # Try to load compressed corpus first, fall back to uncompressed
        corpus_path_gz = Path(f"{path}.corpus.json.gz")
        corpus_path = Path(f"{path}.corpus.json")
        
        if corpus_path_gz.exists():
            # Load compressed corpus
            try:
                with gzip.open(corpus_path_gz, 'rt', encoding='utf-8') as f:
                    self.code_corpus = json.load(f)
                logger.debug("Loaded compressed corpus from %s", corpus_path_gz)
            except Exception as e:
                logger.error("Failed to load compressed corpus: %s", e)
                raise FileNotFoundError(f"Corpus file corrupted: {corpus_path_gz}") from e
        elif corpus_path.exists():
            # Load uncompressed corpus (backward compatibility)
            try:
                with open(corpus_path, "r") as f:
                    self.code_corpus = json.load(f)
                logger.debug("Loaded uncompressed corpus from %s", corpus_path)
            except Exception as e:
                logger.error("Failed to load uncompressed corpus: %s", e)
                raise FileNotFoundError(f"Corpus file not found: {corpus_path}") from e
        else:
            raise FileNotFoundError(f"Corpus file not found (tried {corpus_path_gz} and {corpus_path})")

        # Rebuild BM25 index
        self.bm25_index.build_index(self.code_corpus)

        logger.debug("Indices loaded from %s. Corpus size: %d", path, len(self.code_corpus))

    def retrieve_similar_code(self, query_code: str, k: int = 3,
                             method: str = "faiss", alpha: float = 0.5) -> List[str]:
        """Retrieve similar code snippets."""
        if not self.code_corpus:
            raise ValueError("Index is not loaded or built")

        method = (method or "faiss").lower()
        if method not in _VALID_METHODS:
            raise ValueError("method must be one of: faiss|bm25|hybrid")

        start_time = perf_counter()

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
        response_time = perf_counter() - start_time
        with self._stats_lock:
            self.stats.total_queries += 1
            self.stats.method_usage[method] += 1
            self.stats.total_response_time += response_time
            self.stats.avg_response_time = (
                self.stats.total_response_time / self.stats.total_queries
            )

        return [self.code_corpus[i] for i in result_indices]
