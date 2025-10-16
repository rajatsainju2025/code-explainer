"""FAISS index management for vector search."""

import logging
from pathlib import Path
from typing import Any, List, Optional

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

    def __init__(self, model: SentenceTransformer, batch_size: int = 32):
        if not HAS_FAISS:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")

        self.model = model
        self.batch_size = batch_size
        self.index: Optional[Any] = None

    def build_index(self, codes: List[str]) -> None:
        """Build FAISS index from code snippets."""
        logger.info(f"Building FAISS index for {len(codes)} code snippets...")

        # Encode codes into vectors with batch processing
        embeddings_list = []

        for i in range(0, len(codes), self.batch_size):
            batch_codes = codes[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch_codes, show_progress_bar=False)
            embeddings_list.append(batch_embeddings)

        embeddings = np.concatenate(embeddings_list, axis=0)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Build FAISS index
        d = int(embeddings.shape[1])
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)

        logger.info(f"FAISS index built successfully. Index size: {self.index.ntotal}")

    def search(self, query_code: str, k: int) -> tuple:
        """Search for similar codes using FAISS."""
        if self.index is None:
            raise ValueError("FAISS index is not loaded.")

        query_embedding = self.model.encode([query_code])
        distances, indices = self.index.search(
            np.array(query_embedding, dtype=np.float32),
            min(k, self.index.ntotal)
        )
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