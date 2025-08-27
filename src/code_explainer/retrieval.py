"""Code retrieval for Retrieval-Augmented Generation (RAG).

Supports FAISS vector search, BM25 lexical search, and hybrid fusion.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore

logger = logging.getLogger(__name__)


class CodeRetriever:
    """Handles building and querying a FAISS index for code retrieval.

    Supports FAISS vector search, BM25 lexical search, and hybrid fusion.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model: Optional[Any] = None):
        self.model_name = model_name
        self.model = model or SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.code_corpus: List[str] = []
        self._bm25 = None

    def build_index(self, codes: List[str], save_path: Optional[str] = None) -> None:
        logger.info(f"Building index for {len(codes)} code snippets...")
        self.code_corpus = codes

        # Encode codes into vectors
        embeddings = self.model.encode(codes, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Build FAISS index
        d = int(embeddings.shape[1])
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)  # type: ignore

        logger.info(f"Index built successfully. Index size: {self.index.ntotal}")

        if save_path:
            self.save_index(save_path)

    def save_index(self, path: str) -> None:
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
        self._bm25 = None

    # --- BM25 helpers ---
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is not installed; install rank-bm25 to use BM25 retrieval")
        if not self.code_corpus:
            raise ValueError("No corpus loaded to build BM25 index")
        tokenized_corpus = [self._tokenize(code) for code in self.code_corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        return [t for t in re.split(r"[^A-Za-z0-9_]+", text) if t]

    def retrieve_similar_code(
        self,
        query_code: str,
        k: int = 3,
        method: str = "faiss",
        alpha: float = 0.5,
    ) -> List[str]:
        if not self.code_corpus:
            raise ValueError("Index/corpus is not loaded or built.")

        method = (method or "faiss").lower()
        if method not in {"faiss", "bm25", "hybrid"}:
            raise ValueError("method must be one of: faiss|bm25|hybrid")

        faiss_scores: Dict[int, float] = {}
        bm25_scores: Dict[int, float] = {}

        # FAISS search
        if method in {"faiss", "hybrid"}:
            if self.index is None:
                raise ValueError("FAISS index is not loaded.")
            query_embedding = self.model.encode([query_code])
            distances, indices = self.index.search(  # type: ignore
                np.array(query_embedding, dtype=np.float32), min(k, len(self.code_corpus))
            )
            for d, i in zip(distances[0], indices[0]):
                sim = 1.0 / (1.0 + float(d))
                faiss_scores[int(i)] = sim

        # BM25 search
        if method in {"bm25", "hybrid"}:
            self._ensure_bm25()
            assert self._bm25 is not None
            tokenized_query = self._tokenize(query_code)
            scores = self._bm25.get_scores(tokenized_query)
            top_idx = np.argsort(scores)[::-1][: min(k, len(scores))]
            for i in top_idx:
                bm25_scores[int(i)] = float(scores[int(i)])

        # Decide results
        if method == "faiss":
            ranked = sorted(faiss_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        elif method == "bm25":
            ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        else:  # hybrid
            if bm25_scores:
                bm_min = min(bm25_scores.values())
                bm_max = max(bm25_scores.values())
                bm_range = (bm_max - bm_min) or 1.0
                bm25_norm = {i: (s - bm_min) / bm_range for i, s in bm25_scores.items()}
            else:
                bm25_norm = {}
            candidates = set(faiss_scores) | set(bm25_norm)
            fused: List[Tuple[int, float]] = []
            for i in candidates:
                s_f = faiss_scores.get(i, 0.0)
                s_b = bm25_norm.get(i, 0.0)
                fused.append((i, alpha * s_f + (1 - alpha) * s_b))
            ranked = sorted(fused, key=lambda x: x[1], reverse=True)[:k]

        return [self.code_corpus[i] for i, _ in ranked]
