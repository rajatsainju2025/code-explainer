"""Cross-encoder reranker for improving retrieval relevance."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    CrossEncoder = None

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval relevance."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required for reranking. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        try:
            if CrossEncoder is not None:
                self.model = CrossEncoder(self.model_name)
                logger.info(f"Initialized cross-encoder: {self.model_name}")
            else:
                raise ImportError("CrossEncoder not available")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder {self.model_name}: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder.
        
        Args:
            query: Query text
            candidates: List of candidate documents with 'content' field
            top_k: Number of top candidates to return
            score_threshold: Minimum relevance score threshold
            
        Returns:
            Reranked candidates with relevance scores
        """
        if not candidates:
            return []
        
        if self.model is None:
            logger.warning("Cross-encoder model not initialized, returning original candidates")
            return candidates[:top_k] if top_k else candidates
        
        try:
            # Prepare query-candidate pairs
            pairs = []
            for candidate in candidates:
                content = candidate.get('content', '')
                if isinstance(content, str):
                    pairs.append([query, content])
                else:
                    pairs.append([query, str(content)])
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Combine candidates with scores
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                score = float(scores[i])
                if score >= score_threshold:
                    enhanced_candidate = candidate.copy()
                    enhanced_candidate['rerank_score'] = score
                    scored_candidates.append(enhanced_candidate)
            
            # Sort by relevance score (descending)
            scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top-k results
            result = scored_candidates[:top_k] if top_k else scored_candidates
            
            logger.info(f"Reranked {len(candidates)} candidates, returned {len(result)} above threshold {score_threshold}")
            return result
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:top_k] if top_k else candidates


class MaximalMarginalRelevance:
    """Maximal Marginal Relevance (MMR) for diversified retrieval."""
    
    def __init__(self, lambda_param: float = 0.5):
        """Initialize MMR.
        
        Args:
            lambda_param: Trade-off parameter between relevance and diversity (0-1)
        """
        self.lambda_param = lambda_param
    
    def select(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Select diverse candidates using MMR.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            candidates: List of candidate documents
            top_k: Number of candidates to select
            
        Returns:
            Selected diverse candidates
        """
        if not candidates or len(candidates) != len(candidate_embeddings):
            return candidates[:top_k]
        
        try:
            # Convert to numpy arrays
            query_emb = np.array(query_embedding)
            cand_embs = np.array(candidate_embeddings)
            
            # Compute query-candidate similarities
            query_similarities = self._cosine_similarity(query_emb, cand_embs)
            if isinstance(query_similarities, float):
                query_similarities = np.array([query_similarities])
            elif not isinstance(query_similarities, np.ndarray):
                query_similarities = np.array(query_similarities)
            
            selected_indices = []
            remaining_indices = list(range(len(candidates)))
            
            # Select first candidate with highest query similarity
            if remaining_indices:
                remaining_sims = query_similarities[remaining_indices]
                best_local_idx = np.argmax(remaining_sims)
                best_idx = remaining_indices[best_local_idx]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # Select remaining candidates using MMR
            while len(selected_indices) < top_k and remaining_indices:
                mmr_scores = []
                
                for idx in remaining_indices:
                    # Relevance score
                    relevance = query_similarities[idx]
                    
                    # Diversity score (max similarity with selected candidates)
                    if selected_indices:
                        similarities_to_selected = [
                            self._cosine_similarity(cand_embs[idx], cand_embs[sel_idx])
                            for sel_idx in selected_indices
                        ]
                        max_similarity = max(similarities_to_selected)
                    else:
                        max_similarity = 0.0
                    
                    # MMR score
                    mmr_score = (self.lambda_param * relevance - 
                               (1 - self.lambda_param) * max_similarity)
                    mmr_scores.append(mmr_score)
                
                # Select candidate with highest MMR score
                best_mmr_idx = np.argmax(mmr_scores)
                best_candidate_idx = remaining_indices[best_mmr_idx]
                selected_indices.append(best_candidate_idx)
                remaining_indices.remove(best_candidate_idx)
            
            # Return selected candidates with MMR scores
            result = []
            for i, idx in enumerate(selected_indices):
                candidate = candidates[idx].copy()
                candidate['mmr_rank'] = i + 1
                candidate['query_similarity'] = float(query_similarities[idx])
                result.append(candidate)
            
            logger.info(f"Selected {len(result)} diverse candidates using MMR (λ={self.lambda_param})")
            return result
            
        except Exception as e:
            logger.error(f"MMR selection failed: {e}")
            return candidates[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.ndim == 1 and b.ndim == 2:
            # a is 1D, b is 2D (multiple vectors)
            dot_product = np.dot(b, a)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b, axis=1)
            return dot_product / (norm_a * norm_b + 1e-8)
        elif a.ndim == 1 and b.ndim == 1:
            # Both are 1D vectors
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return dot_product / (norm_a * norm_b + 1e-8)
        else:
            raise ValueError(f"Unsupported vector dimensions: a.shape={a.shape}, b.shape={b.shape}")


def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Optional[CrossEncoderReranker]:
    """Create a reranker instance with error handling.
    
    Args:
        model_name: Name of the cross-encoder model
        
    Returns:
        Reranker instance or None if creation fails
    """
    try:
        return CrossEncoderReranker(model_name)
    except Exception as e:
        logger.warning(f"Failed to create reranker: {e}")
        return None


def create_mmr(lambda_param: float = 0.5) -> MaximalMarginalRelevance:
    """Create an MMR instance.
    
    Args:
        lambda_param: Trade-off parameter between relevance and diversity
        
    Returns:
        MMR instance
    """
    return MaximalMarginalRelevance(lambda_param)
