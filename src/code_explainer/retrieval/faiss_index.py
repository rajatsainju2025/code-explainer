"""FAISS index management for vector search."""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Manages FAISS index for vector similarity search."""
    
    # Use __slots__ to reduce memory overhead
    # Allow instance dict for test-time monkeypatching while keeping most
    # attributes in slots to reduce memory overhead in production.
    __slots__ = ('model', 'batch_size', 'index', '_dimension', '_query_cache', 
                 '_cache_max_size', '__dict__')

    def __init__(self, model: SentenceTransformer, batch_size: int = 32):
        if not HAS_FAISS:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")

        self.model = model
        self.batch_size = batch_size
        self.index: Optional[Any] = None
        self._dimension: Optional[int] = None
        self._query_cache: OrderedDict = OrderedDict()  # LRU cache
        self._cache_max_size = 256  # Increased from 200 for better hit rate

    def build_index(self, codes: List[str], use_ivf: Optional[bool] = None, nlist: int = 100) -> None:
        """Build FAISS index from code snippets.
        
        Args:
            codes: List of code snippets to index
            use_ivf: Use IVF index for faster search on large datasets.
                     None (default) = auto-select based on corpus size (>=5 000).
            nlist: Number of clusters for IVF index
        """
        num_codes = len(codes)
        
        # Adaptive IVF threshold - use IVF for datasets > 5k instead of 1k
        if use_ivf is None:
            use_ivf = num_codes >= 5000
        
        logger.info("Building FAISS index for %d code snippets (IVF: %s)...", num_codes, use_ivf)

        # Use sentence-transformers batch encoding directly (more efficient)
        embeddings = self.model.encode(
            codes, 
            batch_size=self.batch_size,
            show_progress_bar=num_codes > 1000,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Store dimension for later use
        self._dimension = embeddings.shape[1]

        # Build FAISS index - use IVF for large datasets
        if use_ivf and num_codes >= 1000:
            # IVF index for faster search on large datasets
            quantizer = faiss.IndexFlatIP(self._dimension)
            # Use sqrt rule: nlist = sqrt(num_codes) for optimal clustering
            optimal_nlist = int(np.sqrt(num_codes))
            actual_nlist = min(max(optimal_nlist, 50), num_codes // 10)  # Between 50 and num_codes/10
            
            self.index = faiss.IndexIVFFlat(quantizer, self._dimension, actual_nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
            
            # Adaptive nprobe: search more clusters for better recall on larger datasets
            self.index.nprobe = min(max(10, actual_nlist // 20), 50)  # Between 10 and 50
            logger.info("Built IVF index with %d clusters (nprobe: %d)", actual_nlist, self.index.nprobe)
        else:
            # Flat index for smaller datasets or maximum accuracy
            self.index = faiss.IndexFlatIP(self._dimension)  # Use inner product for normalized vectors
            self.index.add(embeddings)

        # Clear query cache since index changed
        self._query_cache.clear()
        
        logger.info("FAISS index built successfully. Index size: %d", self.index.ntotal)

    def search(self, query_code: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar codes using FAISS."""
        if self.index is None:
            from ..exceptions import ResourceError
            raise ResourceError("FAISS index is not loaded.", resource_type="faiss_index")

        # Check cache for repeated queries
        cache_key = (query_code, k)
        if cache_key in self._query_cache:
            # Move to end (most recently used)
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key]

        query_embedding = self.model.encode(
            [query_code], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        
        k_actual = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k_actual)
        
        # Cache result with LRU eviction
        self._query_cache[cache_key] = (distances, indices)
        if len(self._query_cache) > self._cache_max_size:
            self._query_cache.popitem(last=False)  # Remove oldest
        
        return distances, indices

    def save_index(self, path: str) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            from ..exceptions import ResourceError
            raise ResourceError("Index has not been built yet.", resource_type="faiss_index")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p))
        logger.info("FAISS index saved to %s", path)

    def load_index(self, path: str) -> None:
        """Load FAISS index from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        # Wrap faiss read_index in a safe loader that validates the index
        try:
            self.index = faiss.read_index(str(p))
        except Exception as e:
            logger.error("Failed to read FAISS index from %s: %s", path, e)
            raise

        # Sanity check: ensure index has a non-negative number of vectors
        try:
            ntotal = int(getattr(self.index, 'ntotal', 0) or 0)
        except Exception:
            ntotal = 0

        if ntotal <= 0:
            logger.warning("Loaded FAISS index has zero entries: %s", path)

        logger.info("FAISS index loaded from %s. Index size: %d", path, ntotal)

    def load_index_safe(self, path: str) -> None:
        """Compatibility wrapper that loads and validates a FAISS index.

        This helper is callable by tests that want to ensure the index file is
        present and contains vectors. It returns without raising if the index
        is loaded but empty (caller can decide to rebuild).
        """
        try:
            self.load_index(path)
        except FileNotFoundError:
            # Bubble up missing file as-is
            raise
        except Exception:
            # For other errors, log and re-raise as ResourceError for consistency
            from ..exceptions import ResourceError
            raise ResourceError("Failed to load FAISS index", resource_type="faiss_index")