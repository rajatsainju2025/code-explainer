"""Enhanced code retrieval with FAISS, BM25, hybrid search, reranking, and MMR."""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib
import time
import re
import threading
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None

from .reranker import create_reranker, create_mmr

logger = logging.getLogger(__name__)


class CodeRetriever:
    """Handles building and querying a FAISS index for code retrieval.

    Supports FAISS vector search, BM25 lexical search, and hybrid fusion.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model: Optional[Any] = None):
        """Initialize the code retriever.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = model if model is not None else SentenceTransformer(model_name)

        # Core data structures
        self.code_corpus: List[str] = []
        self.index: Optional[Any] = None  # FAISS index
        self._bm25: Optional[Any] = None  # BM25 index

        # Advanced features
        self.reranker = create_reranker()  # Cross-encoder reranker
        self.mmr = create_mmr(lambda_param=0.5)  # MMR for diversity

        # Statistics
        self.retrieval_stats = {
            "total_queries": 0,
            "method_usage": {"faiss": 0, "bm25": 0, "hybrid": 0},
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "rerank_usage": 0,
            "mmr_usage": 0
        }
        self._stats_lock = threading.Lock()  # Thread-safe statistics updates

    def build_index(self, codes: List[str], save_path: Optional[str] = None) -> None:
        logger.info(f"Building index for {len(codes)} code snippets...")
        self.code_corpus = codes

        if not HAS_FAISS or faiss is None:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")

        # Encode codes into vectors with batch processing for better performance
        batch_size = 32  # Optimal batch size for most models
        embeddings_list = []

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
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

        logger.info(f"Index built successfully. Index size: {self.index.ntotal}")

        if save_path:
            self.save_index(save_path)

    def save_index(self, path: str) -> None:
        if self.index is None:
            raise ValueError("Index has not been built yet.")
        if not HAS_FAISS or faiss is None:
            raise ImportError("FAISS is not available")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(p))

        # Save code corpus
        with open(f"{path}.corpus.json", "w") as f:
            json.dump(self.code_corpus, f)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        if not HAS_FAISS or faiss is None:
            raise ImportError("FAISS is not available")

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        # Load FAISS index
        self.index = faiss.read_index(str(p))

        corpus_path = Path(f"{path}.corpus.json")
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        with open(corpus_path, "r") as f:
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
        tokenized_corpus = [list(self._tokenize(code)) for code in self.code_corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _tokenize(text: str) -> Tuple[str, ...]:
        """Tokenize text for BM25 indexing with caching."""
        # Pre-compile regex pattern for better performance
        token_pattern = re.compile(r"[^A-Za-z0-9_]+")
        tokens = [t for t in token_pattern.split(text) if t]
        return tuple(tokens)  # Return tuple for hashability (required for lru_cache)

    def retrieve_similar_code(
        self,
        query_code: str,
        k: int = 3,
        method: str = "faiss",
        alpha: float = 0.5,
    ) -> List[str]:
        if not self.code_corpus:
            raise ValueError("Index is not loaded or built")

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
            tokenized_query = list(self._tokenize(query_code))
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
        """Enhanced retrieval with reranking and MMR for diversity.

        Args:
            query_code: Code query
            k: Number of results to return
            method: Retrieval method (faiss, bm25, hybrid)
            alpha: Weighting for hybrid search
            use_reranker: Whether to use cross-encoder reranking
            use_mmr: Whether to use MMR for diversity
            rerank_top_k: Number of candidates to rerank
            mmr_lambda: MMR trade-off parameter

        Returns:
            List of enhanced result dictionaries with scores and metadata
        """
        start_time = time.time()

        # Update statistics
        self.retrieval_stats["total_queries"] += 1
        self.retrieval_stats["method_usage"][method] += 1

        if not self.code_corpus:
            raise ValueError("Index/corpus is not loaded or built.")

        method = (method or "faiss").lower()
        if method not in {"faiss", "bm25", "hybrid"}:
            raise ValueError("method must be one of: faiss|bm25|hybrid")

        # Get initial candidates (more than k if we're reranking)
        initial_k = max(k, rerank_top_k if use_reranker else k)

        faiss_scores: Dict[int, float] = {}
        bm25_scores: Dict[int, float] = {}

        # FAISS search
        if method in {"faiss", "hybrid"}:
            if self.index is None:
                raise ValueError("FAISS index is not loaded.")
            query_embedding = self.model.encode([query_code])
            distances, indices = self.index.search(
                np.array(query_embedding, dtype=np.float32),
                min(initial_k, len(self.code_corpus))
            )
            for d, i in zip(distances[0], indices[0]):
                sim = 1.0 / (1.0 + float(d))
                faiss_scores[int(i)] = sim

        # BM25 search
        if method in {"bm25", "hybrid"}:
            self._ensure_bm25()
            assert self._bm25 is not None
            tokenized_query = list(self._tokenize(query_code))
            scores = self._bm25.get_scores(tokenized_query)
            top_idx = np.argsort(scores)[::-1][:min(initial_k, len(scores))]
            for i in top_idx:
                bm25_scores[int(i)] = float(scores[int(i)])

        # Combine scores based on method
        if method == "faiss":
            ranked = sorted(faiss_scores.items(), key=lambda x: x[1], reverse=True)[:initial_k]
        elif method == "bm25":
            ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:initial_k]
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
            ranked = sorted(fused, key=lambda x: x[1], reverse=True)[:initial_k]

        # Convert to candidate dictionaries
        candidates = []
        for i, score in ranked:
            candidate = {
                "content": self.code_corpus[i],
                "index": i,
                "initial_score": score,
                "method": method
            }
            candidates.append(candidate)

        # Apply reranking if requested
        if use_reranker and self.reranker is not None and candidates:
            try:
                candidates = self.reranker.rerank(
                    query_code,
                    candidates,
                    top_k=k if not use_mmr else min(k * 2, len(candidates))
                )
                self.retrieval_stats["rerank_usage"] += 1
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        # Apply MMR for diversity if requested
        if use_mmr and self.mmr is not None and len(candidates) > 1:
            try:
                # Get embeddings for MMR
                query_embedding = self.model.encode([query_code])[0]
                candidate_contents = [c["content"] for c in candidates]
                candidate_embeddings = self.model.encode(candidate_contents)

                # Convert to list of arrays for MMR
                candidate_embedding_list = [candidate_embeddings[i] for i in range(len(candidate_embeddings))]

                # Update MMR lambda if provided
                if mmr_lambda != 0.5:
                    self.mmr.lambda_param = mmr_lambda

                candidates = self.mmr.select(
                    query_embedding,
                    candidate_embedding_list,
                    candidates,
                    top_k=k
                )
                self.retrieval_stats["mmr_usage"] += 1
            except Exception as e:
                logger.warning(f"MMR selection failed: {e}")
                candidates = candidates[:k]
        else:
            candidates = candidates[:k]

        # Update timing statistics
        response_time = time.time() - start_time
        self.retrieval_stats["total_response_time"] += response_time
        self.retrieval_stats["avg_response_time"] = (
            self.retrieval_stats["total_response_time"] / self.retrieval_stats["total_queries"]
        )

        return candidates
