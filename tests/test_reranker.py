"""Tests for reranker and MMR functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.code_explainer.reranker import (
    CrossEncoderReranker,
    MaximalMarginalRelevance,
    create_reranker,
    create_mmr
)


class TestCrossEncoderReranker:
    """Test cross-encoder reranker functionality."""

    def test_create_reranker_without_sentence_transformers(self):
        """Test reranker creation when sentence-transformers is not available."""
        with patch('src.code_explainer.reranker.HAS_SENTENCE_TRANSFORMERS', False):
            reranker = create_reranker()
            assert reranker is None

    @patch('src.code_explainer.reranker.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('src.code_explainer.reranker.CrossEncoder')
    def test_reranker_initialization(self, mock_cross_encoder):
        """Test reranker initialization with mocked CrossEncoder."""
        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker("test-model")
        assert reranker.model == mock_model
        assert reranker.model_name == "test-model"

    @patch('src.code_explainer.reranker.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('src.code_explainer.reranker.CrossEncoder')
    def test_rerank_with_scores(self, mock_cross_encoder):
        """Test reranking with mock scores."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.3, 0.9, 0.1])
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker("test-model")

        candidates = [
            {"content": "def add(a, b): return a + b"},
            {"content": "def sub(a, b): return a - b"},
            {"content": "def multiply(a, b): return a * b"},
            {"content": "def divide(a, b): return a / b"}
        ]

        result = reranker.rerank("arithmetic functions", candidates, top_k=2)

        assert len(result) == 2
        assert result[0]["rerank_score"] == 0.9  # Highest score
        assert result[1]["rerank_score"] == 0.8  # Second highest
        assert result[0]["content"] == "def multiply(a, b): return a * b"

    @patch('src.code_explainer.reranker.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('src.code_explainer.reranker.CrossEncoder')
    def test_rerank_with_threshold(self, mock_cross_encoder):
        """Test reranking with score threshold."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.2, 0.8, 0.1, 0.9])
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker("test-model")

        candidates = [
            {"content": "code1"},
            {"content": "code2"},
            {"content": "code3"},
            {"content": "code4"}
        ]

        result = reranker.rerank("query", candidates, score_threshold=0.5)

        assert len(result) == 2  # Only 2 candidates above threshold
        assert all(r["rerank_score"] >= 0.5 for r in result)

    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidates."""
        with patch('src.code_explainer.reranker.HAS_SENTENCE_TRANSFORMERS', True):
            with patch('src.code_explainer.reranker.CrossEncoder'):
                reranker = CrossEncoderReranker("test-model")
                result = reranker.rerank("query", [])
                assert result == []


class TestMaximalMarginalRelevance:
    """Test MMR functionality."""

    def test_mmr_initialization(self):
        """Test MMR initialization."""
        mmr = MaximalMarginalRelevance(lambda_param=0.7)
        assert mmr.lambda_param == 0.7

    def test_mmr_selection_basic(self):
        """Test basic MMR selection."""
        mmr = MaximalMarginalRelevance(lambda_param=0.5)

        # Mock query and candidate embeddings
        query_embedding = np.array([1.0, 0.0, 0.0])
        candidate_embeddings = [
            np.array([0.9, 0.1, 0.0]),  # High similarity to query
            np.array([0.8, 0.2, 0.0]),  # Medium similarity to query
            np.array([0.0, 0.0, 1.0]),  # Low similarity to query
        ]

        candidates = [
            {"content": "similar1"},
            {"content": "similar2"},
            {"content": "different"}
        ]

        result = mmr.select(query_embedding, candidate_embeddings, candidates, top_k=2)

        assert len(result) == 2
        assert all("mmr_rank" in r for r in result)
        assert all("query_similarity" in r for r in result)

        # First result should be most similar to query
        assert result[0]["mmr_rank"] == 1
        assert result[0]["content"] == "similar1"

    def test_mmr_empty_candidates(self):
        """Test MMR with empty candidates."""
        mmr = MaximalMarginalRelevance()

        query_embedding = np.array([1.0, 0.0])
        result = mmr.select(query_embedding, [], [], top_k=5)

        assert result == []

    def test_mmr_mismatched_lengths(self):
        """Test MMR with mismatched candidate and embedding lengths."""
        mmr = MaximalMarginalRelevance()

        query_embedding = np.array([1.0, 0.0])
        candidate_embeddings = [np.array([0.5, 0.5])]
        candidates = [{"content": "code1"}, {"content": "code2"}]  # Mismatch

        result = mmr.select(query_embedding, candidate_embeddings, candidates, top_k=2)
        assert len(result) <= 2  # Should handle gracefully

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        mmr = MaximalMarginalRelevance()

        # Test 1D vectors
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        similarity = mmr._cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 1e-6  # Orthogonal vectors

        # Test identical vectors
        similarity = mmr._cosine_similarity(a, a)
        assert abs(similarity - 1.0) < 1e-6

        # Test 1D vs 2D
        b_multi = np.array([[1.0, 0.0], [0.0, 1.0]])
        similarities = mmr._cosine_similarity(a, b_multi)
        if isinstance(similarities, np.ndarray):
            assert len(similarities) == 2
            assert abs(similarities[0] - 1.0) < 1e-6
            assert abs(similarities[1] - 0.0) < 1e-6
        else:
            # Single similarity score case
            assert isinstance(similarities, (float, np.floating))


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_reranker_success(self):
        """Test successful reranker creation."""
        with patch('src.code_explainer.reranker.HAS_SENTENCE_TRANSFORMERS', True):
            with patch('src.code_explainer.reranker.CrossEncoder'):
                reranker = create_reranker("test-model")
                assert reranker is not None
                assert reranker.model_name == "test-model"

    def test_create_reranker_failure(self):
        """Test reranker creation failure."""
        with patch('src.code_explainer.reranker.CrossEncoderReranker') as mock_cls:
            mock_cls.side_effect = Exception("Model loading failed")

            reranker = create_reranker()
            assert reranker is None

    def test_create_mmr(self):
        """Test MMR creation."""
        mmr = create_mmr(lambda_param=0.3)
        assert mmr is not None
        assert mmr.lambda_param == 0.3


if __name__ == "__main__":
    pytest.main([__file__])
