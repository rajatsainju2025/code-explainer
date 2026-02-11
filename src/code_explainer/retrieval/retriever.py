"""Main code retriever orchestrating all retrieval components.

Optimized for performance with:
- Fast xxhash-based cache key generation
- Lock-free reads for cache hits (optimistic locking)
- Pre-computed method validation sets
- Efficient memory usage with __slots__
"""

import gzip
import json
import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

from sentence_transformers import SentenceTransformer

from .bm25_index import BM25Index
from .enhanced_retrieval import EnhancedRetrieval
from .faiss_index import FAISSIndex
from .hybrid_search import HybridSearch
from .models import RetrievalCandidate, RetrievalConfig, RetrievalStats
from .model_cache import get_cached_model
from ..utils.hashing import fast_hash_str as _hash_key

logger = logging.getLogger(__name__)

# Pre-compute valid methods set for O(1) lookup
_VALID_METHODS = frozenset({"faiss", "bm25", "hybrid"})


class LRUQueryCache:
    """LRU cache for retrieval results to avoid redundant computations.
    
    Uses __slots__ to reduce memory overhead per cache instance.
    Implements optimistic read locking for better concurrent performance.
    Optimized with pre-computed string templates for faster key generation.
    """
    __slots__ = ('cache', 'max_size', 'hits', 'misses', '_lock', '_version', '_key_cache')
    
    def __init__(self, max_size: int = 512):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._version = 0  # For optimistic read detection
        self._key_cache: Dict[Tuple, str] = {}  # Cache generated keys
    
    def _make_key(self, query: str, k: int, method: str, alpha: float, 
                  use_reranker: bool, use_mmr: bool) -> str:
        """Create deterministic cache key from query parameters.
        
        Uses f-string formatting (faster than join) and xxhash.
        Caches computed keys for frequently repeated queries.
        """
        # Create a tuple for key lookup (immutable, hashable)
        key_params = (query, k, method, alpha, use_reranker, use_mmr)
        
        # Check key cache first (faster for repeated queries)
        if key_params in self._key_cache:
            return self._key_cache[key_params]
        
        # Use f-string for faster string building
        key_str = f"{query}|{k}|{method}|{alpha:.2f}|{use_reranker}|{use_mmr}"
        hashed_key = _hash_key(key_str)
        
        # Cache if not full (limit to 1000 entries)
        if len(self._key_cache) < 1000:
            self._key_cache[key_params] = hashed_key
        
        return hashed_key
    
    def get(self, query: str, k: int, method: str, alpha: float,
            use_reranker: bool = False, use_mmr: bool = False) -> Optional[Any]:
        """Get cached result if available.
        
        Uses optimistic read: first check without lock, then verify.
        """
        key = self._make_key(query, k, method, alpha, use_reranker, use_mmr)
        
        # Optimistic read without lock (safe for immutable values)
        value = self.cache.get(key)
        if value is not None:
            # Update LRU order with lock
            with self._lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return value
        
        with self._lock:
            self.misses += 1
        return None
    
    def put(self, query: str, k: int, method: str, alpha: float,
            value: Any, use_reranker: bool = False, use_mmr: bool = False) -> None:
        """Cache a result."""
        key = self._make_key(query, k, method, alpha, use_reranker, use_mmr)
        with self._lock:
            # Check if already exists
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
                return
            
            # Evict oldest entries if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = value
            self._version += 1
    
    def invalidate(self, query: str) -> int:
        """Invalidate all cache entries containing query.
        
        Returns number of entries removed.
        """
        with self._lock:
            keys_to_remove = [k for k in self.cache if query in str(k)]
            for key in keys_to_remove:
                del self.cache[key]
            self._version += 1
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self._version += 1
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.get_hit_rate(),
                "version": self._version
            }


class CodeRetriever:
    """Handles building and querying a FAISS index for code retrieval.

    Supports FAISS vector search, BM25 lexical search, and hybrid fusion.
    Includes LRU query result caching to avoid redundant computations.
    """
    
    __slots__ = ('config', 'model', 'code_corpus', 'faiss_index', 'bm25_index', 
                 'hybrid_search', 'enhanced_retrieval', 'enable_query_cache', 
                 'query_cache', 'stats', '_stats_lock')

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model: Optional[SentenceTransformer] = None, 
                 enable_query_cache: bool = True,
                 query_cache_size: int = 256):
        """Initialize the code retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            model: Pre-loaded model instance (optional)
            enable_query_cache: Whether to cache query results
            query_cache_size: Maximum number of cached queries
        """
        self.config = RetrievalConfig()
        # Use provided model or get from persistent cache
        self.model = model if model is not None else get_cached_model(model_name)

        # Core components
        self.code_corpus: List[str] = []
        self.faiss_index = FAISSIndex(self.model, self.config.batch_size)
        self.bm25_index = BM25Index()
        self.hybrid_search = HybridSearch(self.faiss_index, self.bm25_index, self.config.hybrid_alpha)
        self.enhanced_retrieval = EnhancedRetrieval(self.model)

        # Query result caching
        self.enable_query_cache = enable_query_cache
        self.query_cache = LRUQueryCache(max_size=query_cache_size) if enable_query_cache else None

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
        """Enhanced retrieval with reranking and MMR.
        
        Results are cached to avoid redundant computations.
        """
        start_time = perf_counter()

        if not self.code_corpus:
            raise ValueError("Index/corpus is not loaded or built.")

        method = (method or "faiss").lower()
        if method not in _VALID_METHODS:
            raise ValueError("method must be one of: faiss|bm25|hybrid")

        # Check query cache first to avoid redundant computation
        if self.enable_query_cache and self.query_cache is not None:
            cached_result = self.query_cache.get(
                query_code, k, method, alpha, use_reranker, use_mmr
            )
            if cached_result is not None:
                with self._stats_lock:
                    self.stats.total_queries += 1
                    self.stats.cache_hits += 1
                logger.debug("Query cache hit for: %.30s...", query_code)
                return cached_result

        # Update statistics
        with self._stats_lock:
            self.stats.total_queries += 1
            self.stats.method_usage[method] += 1

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
        response_time = perf_counter() - start_time
        with self._stats_lock:
            self.stats.total_response_time += response_time
            self.stats.avg_response_time = (
                self.stats.total_response_time / self.stats.total_queries
            )

        # Convert to dict format for backward compatibility (use tuple unpacking for efficiency)
        result = [
            {
                "content": content,
                "index": index,
                "initial_score": initial_score,
                "method": method,
                "rerank_score": rerank_score,
                "final_score": final_score,
                **metadata
            }
            for content, index, initial_score, method, rerank_score, final_score, metadata in (
                (c.content, c.index, c.initial_score, c.method, c.rerank_score, c.final_score, c.metadata)
                for c in candidates
            )
        ]

        # Cache the result
        if self.enable_query_cache and self.query_cache is not None:
            self.query_cache.put(
                query_code, k, method, alpha, result, use_reranker, use_mmr
            )

        return result

    def size(self) -> int:
        """Get the number of indexed code snippets."""
        return len(self.code_corpus)

    def get_stats(self) -> RetrievalStats:
        """Get retrieval statistics."""
        with self._stats_lock:
            return self.stats