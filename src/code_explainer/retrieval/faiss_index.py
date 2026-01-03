"""FAISS index management for vector search."""

import logging
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
    __slots__ = ('model', 'batch_size', 'index', '_dimension', '_query_cache')

    def __init__(self, model: SentenceTransformer, batch_size: int = 32):
        if not HAS_FAISS:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")

        self.model = model
        self.batch_size = batch_size
        self.index: Optional[Any] = None
        self._dimension: Optional[int] = None
        self._query_cache: dict = {}  # Small cache for repeated queries

    def build_index(self, codes: List[str], use_ivf: bool = False, nlist: int = 100) -> None:
        """Build FAISS index from code snippets.
        
        Args:
            codes: List of code snippets to index
            use_ivf: Use IVF index for faster search on large datasets (>10k)
            nlist: Number of clusters for IVF index
        """
        logger.info(f"Building FAISS index for {len(codes)} code snippets...")

        # Use sentence-transformers batch encoding directly (more efficient)
        embeddings = self.model.encode(
            codes, 
            batch_size=self.batch_size,
            show_progress_bar=len(codes) > 1000,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Store dimension for later use
        self._dimension = embeddings.shape[1]

        # Build FAISS index - use IVF for large datasets
        if use_ivf and len(codes) >= 1000:
            # IVF index for faster search on large datasets
            quantizer = faiss.IndexFlatIP(self._dimension)
            actual_nlist = min(nlist, len(codes) // 10)  # At least 10 items per cluster
            self.index = faiss.IndexIVFFlat(quantizer, self._dimension, actual_nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(10, actual_nlist)  # Search 10 clusters by default
            logger.info(f"Built IVF index with {actual_nlist} clusters")
        else:
            # Flat index for smaller datasets or maximum accuracy
            self.index = faiss.IndexFlatIP(self._dimension)  # Use inner product for normalized vectors
            self.index.add(embeddings)

        # Clear query cache since index changed
        self._query_cache.clear()
        
        logger.info(f"FAISS index built successfully. Index size: {self.index.ntotal}")

    def search(self, query_code: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar codes using FAISS."""
        if self.index is None:
            raise ValueError("FAISS index is not loaded.")

        # Check cache for repeated queries
        cache_key = (query_code, k)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        query_embedding = self.model.encode(
            [query_code], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        
        k_actual = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k_actual)
        
        # Cache result (limit cache size)
        if len(self._query_cache) > 100:
            self._query_cache.clear()
        self._query_cache[cache_key] = (distances, indices)
        
        return distances, indices

    def search_batch(self, query_codes: List[str], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Batch search for similar codes - more efficient for multiple queries."""
        if self.index is None:
            raise ValueError("FAISS index is not loaded.")

        query_embeddings = self.model.encode(
            query_codes,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        
        k_actual = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embeddings, k_actual)
        
        return distances, indices

    def save_index(self, path: str) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p))
        logger.info(f"FAISS index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load FAISS index from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        self.index = faiss.read_index(str(p))
        logger.info(f"FAISS index loaded from {path}. Index size: {self.index.ntotal}")

    def get_size(self) -> int:
        """Get the number of vectors in the index."""
        return self.index.ntotal if self.index else 0