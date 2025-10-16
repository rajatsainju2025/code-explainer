"""Enhanced retrieval with reranking and MMR."""

import logging
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from ..reranker import create_reranker, create_mmr
from .models import RetrievalCandidate

logger = logging.getLogger(__name__)


class EnhancedRetrieval:
    """Enhanced retrieval with reranking and MMR for diversity."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.reranker = create_reranker()
        self.mmr = create_mmr(lambda_param=0.5)

    def apply_reranking(self, query: str, candidates: List[RetrievalCandidate],
                       top_k: int) -> List[RetrievalCandidate]:
        """Apply cross-encoder reranking to candidates."""
        if not self.reranker or not candidates:
            return candidates

        try:
            # Convert to dict format expected by reranker
            candidate_dicts = [
                {
                    "content": c.content,
                    "index": c.index,
                    "initial_score": c.initial_score,
                    "method": c.method,
                    **c.metadata
                }
                for c in candidates
            ]

            reranked_dicts = self.reranker.rerank(query, candidate_dicts, top_k=top_k)

            # Convert back to RetrievalCandidate
            return [
                RetrievalCandidate(
                    content=d["content"],
                    index=d["index"],
                    initial_score=d.get("initial_score", 0.0),
                    method=d.get("method", "reranked"),
                    rerank_score=d.get("rerank_score"),
                    metadata=d
                )
                for d in reranked_dicts
            ]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates[:top_k]

    def apply_mmr(self, query_embedding: np.ndarray,
                  candidate_embeddings: List[np.ndarray],
                  candidates: List[RetrievalCandidate],
                  top_k: int, lambda_param: float = 0.5) -> List[RetrievalCandidate]:
        """Apply MMR for diversity in results."""
        if not self.mmr or len(candidates) <= 1:
            return candidates[:top_k]

        try:
            # Convert to dict format expected by MMR
            candidate_dicts = [
                {
                    "content": c.content,
                    "index": c.index,
                    "initial_score": c.initial_score,
                    "method": c.method,
                    **c.metadata
                }
                for c in candidates
            ]

            # Update MMR lambda if provided
            if lambda_param != self.mmr.lambda_param:
                self.mmr.lambda_param = lambda_param

            mmr_dicts = self.mmr.select(query_embedding, candidate_embeddings, candidate_dicts, top_k=top_k)

            # Convert back to RetrievalCandidate
            return [
                RetrievalCandidate(
                    content=d["content"],
                    index=d["index"],
                    initial_score=d.get("initial_score", 0.0),
                    method=d.get("method", "mmr"),
                    metadata=d
                )
                for d in mmr_dicts
            ]
        except Exception as e:
            logger.warning(f"MMR selection failed: {e}")
            return candidates[:top_k]

    def enhance_results(self, query: str, candidates: List[RetrievalCandidate],
                       use_reranker: bool = False, use_mmr: bool = False,
                       rerank_top_k: int = 20, mmr_lambda: float = 0.5) -> List[RetrievalCandidate]:
        """Apply enhancement techniques to retrieval results."""
        enhanced_candidates = candidates.copy()

        # Apply reranking
        if use_reranker:
            enhanced_candidates = self.apply_reranking(query, enhanced_candidates, rerank_top_k)

        # Apply MMR for diversity
        if use_mmr and len(enhanced_candidates) > 1:
            query_embedding = self.model.encode([query])[0]
            candidate_contents = [c.content for c in enhanced_candidates]
            candidate_embeddings = self.model.encode(candidate_contents)
            candidate_embedding_list = [candidate_embeddings[i] for i in range(len(candidate_embeddings))]

            enhanced_candidates = self.apply_mmr(
                query_embedding, candidate_embedding_list, enhanced_candidates,
                top_k=len(candidates), lambda_param=mmr_lambda
            )

        return enhanced_candidates