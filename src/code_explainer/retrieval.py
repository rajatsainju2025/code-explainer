"""Code retrieval for Retrieval-Augmented Generation (RAG)."""

import logging
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class CodeRetriever:
    """Handles building and querying a FAISS index for code retrieval."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the code retriever.

        Args:
            model_name: Name of the sentence transformer model to use.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.code_corpus: List[str] = []

    def build_index(self, codes: List[str], save_path: Optional[str] = None) -> None:
        """Build a FAISS index from a list of code snippets.

        Args:
            codes: List of code snippets to index.
            save_path: Optional path to save the FAISS index.
        """
        logger.info(f"Building index for {len(codes)} code snippets...")
        self.code_corpus = codes

        # Encode codes into vectors
        embeddings = self.model.encode(codes, show_progress_bar=True)

        # Build FAISS index
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

        logger.info(f"Index built successfully. Index size: {self.index.ntotal}")

        if save_path:
            self.save_index(save_path)

    def save_index(self, path: str) -> None:
        """Save the FAISS index and code corpus.

        Args:
            path: Path to save the index files.
        """
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(p))

        # Save code corpus
        with open(f"{path}.corpus.json", "w") as f:
            import json

            json.dump(self.code_corpus, f)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load a pre-built FAISS index and code corpus.

        Args:
            path: Path to the FAISS index file.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        self.index = faiss.read_index(str(p))

        corpus_path = Path(f"{path}.corpus.json")
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        with open(corpus_path, "r") as f:
            import json

            self.code_corpus = json.load(f)

        logger.info(f"Index loaded from {path}. Corpus size: {len(self.code_corpus)}")

    def retrieve_similar_code(self, query_code: str, k: int = 3) -> List[str]:
        """Retrieve the most similar code snippets from the index.

        Args:
            query_code: The code snippet to find similar examples for.
            k: The number of similar snippets to retrieve.

        Returns:
            A list of the most similar code snippets.
        """
        if self.index is None or not self.code_corpus:
            raise ValueError("Index is not loaded or built.")

        query_embedding = self.model.encode([query_code])

        # Search the index
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)

        # Retrieve similar codes
        similar_codes = [self.code_corpus[i] for i in indices[0]]

        return similar_codes
