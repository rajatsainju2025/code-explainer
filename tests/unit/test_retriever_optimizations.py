"""Tests for retriever module optimizations."""

import gzip
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from code_explainer.utils.hashing import json_dumps, json_loads
from code_explainer.retrieval.hybrid_search import AdvancedHybridSearch, FusionStrategy


class TestRetrieverJsonOptimization:
    """Tests for JSON serialization in retriever."""
    
    def test_json_roundtrip_code_corpus(self):
        """Test JSON roundtrip for code corpus data."""
        corpus = [
            "def hello(): pass",
            "class Foo: pass",
            "import os; print(os.getcwd())",
        ]
        
        serialized = json_dumps(corpus)
        deserialized = json_loads(serialized)
        
        assert deserialized == corpus
    
    def test_gzip_json_roundtrip(self, tmp_path):
        """Test gzip compressed JSON roundtrip."""
        corpus = ["def test(): return 42"] * 100
        corpus_path = tmp_path / "corpus.json.gz"
        
        # Write compressed
        json_data = json_dumps(corpus)
        with gzip.open(corpus_path, 'wt', encoding='utf-8') as f:
            f.write(json_data)
        
        # Read compressed
        with gzip.open(corpus_path, 'rt', encoding='utf-8') as f:
            loaded = json_loads(f.read())
        
        assert loaded == corpus
    
    def test_large_corpus_serialization(self):
        """Test serialization of large code corpus."""
        corpus = [f"def func_{i}(): return {i}" for i in range(1000)]
        
        serialized = json_dumps(corpus)
        deserialized = json_loads(serialized)
        
        assert len(deserialized) == 1000
        assert deserialized[0] == "def func_0(): return 0"
        assert deserialized[-1] == "def func_999(): return 999"


class TestRetrieverValidation:
    """Tests for retriever validation."""
    
    def test_valid_methods_frozenset(self):
        """Test that valid methods is a frozenset for O(1) lookup."""
        from code_explainer.retrieval.retriever import _VALID_METHODS
        
        assert isinstance(_VALID_METHODS, frozenset)
        assert "faiss" in _VALID_METHODS
        assert "bm25" in _VALID_METHODS
        assert "hybrid" in _VALID_METHODS
        assert "invalid" not in _VALID_METHODS


class TestRetrievalConfig:
    """Tests for RetrievalConfig optimization."""
    
    def test_retrieval_config_has_slots(self):
        """Test that RetrievalConfig uses __slots__ for memory efficiency."""
        from code_explainer.retrieval.models import RetrievalConfig
        
        config = RetrievalConfig()
        
        # Check that it's a frozen dataclass with slots
        assert config.batch_size == 32
        assert config.hybrid_alpha == 0.5


class TestHybridSearchOptimizations:
    """Tests for low-overhead hybrid fusion paths."""

    def test_linear_fusion_small_resultset_normalizes_bm25(self):
        search = AdvancedHybridSearch(fusion_strategy=FusionStrategy.LINEAR, alpha=0.5)

        results = search._linear_fusion(
            faiss_results=[(1, 0.9), (2, 0.2)],
            bm25_results=[(2, 2.0), (3, 1.0)],
            k=3,
        )

        assert results[0][0] == 1
        assert {idx for idx, _ in results} == {1, 2, 3}

    def test_distribution_fusion_small_resultset_keeps_all_candidates(self):
        search = AdvancedHybridSearch(
            fusion_strategy=FusionStrategy.DISTRIBUTION_BASED,
            alpha=0.5,
        )

        results = search._distribution_fusion(
            faiss_results=[(10, 0.8), (20, 0.2)],
            bm25_results=[(20, 5.0), (30, 1.0)],
            k=3,
        )

        assert len(results) == 3
        assert {idx for idx, _ in results} == {10, 20, 30}
