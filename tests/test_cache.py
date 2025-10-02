"""Tests for caching functionality."""

import tempfile
from pathlib import Path

import pytest

from code_explainer.cache import EmbeddingCache, ExplanationCache
from code_explainer.advanced_cache import CacheManager


class TestExplanationCache:
    """Test cases for explanation caching."""

    def test_cache_init(self):
        """Test cache initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ExplanationCache(temp_dir, max_size=10)
            assert cache.cache_dir == Path(temp_dir)
            assert cache.max_size == 10
            assert cache.size() == 0

    def test_cache_put_get(self):
        """Test putting and getting explanations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ExplanationCache(temp_dir)

            code = "def hello(): print('hello')"
            strategy = "vanilla"
            model = "test-model"
            explanation = "This function prints hello"

            # Put explanation
            cache.put(code, strategy, model, explanation)
            assert cache.size() == 1

            # Get explanation
            retrieved = cache.get(code, strategy, model)
            assert retrieved == explanation

    def test_cache_miss(self):
        """Test cache miss for non-existent entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ExplanationCache(temp_dir)

            result = cache.get("def test(): pass", "vanilla", "model")
            assert result is None

    def test_cache_cleanup(self):
        """Test cache cleanup when max size is exceeded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ExplanationCache(temp_dir, max_size=2)

            # Add entries beyond max size
            for i in range(3):
                cache.put(f"def func{i}(): pass", "vanilla", "model", f"explanation {i}")

            # Should only keep 2 entries
            assert cache.size() <= 2

    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ExplanationCache(temp_dir)

            # Add some entries
            cache.put("def a(): pass", "vanilla", "model1", "explanation a")
            cache.put("def b(): pass", "ast_augmented", "model2", "explanation b")

            stats = cache.stats()
            assert stats["size"] == 2
            assert "vanilla" in stats["strategies"]
            assert "ast_augmented" in stats["strategies"]
            assert "model1" in stats["models"]
            assert "model2" in stats["models"]

    def test_cache_clear(self):
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ExplanationCache(temp_dir)

            cache.put("def test(): pass", "vanilla", "model", "explanation")
            assert cache.size() == 1

            cache.clear()
            assert cache.size() == 0


class TestEmbeddingCache:
    """Test cases for embedding caching."""

    def test_embedding_cache_init(self):
        """Test embedding cache initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(temp_dir)
            assert cache.cache_dir == Path(temp_dir)

    def test_embedding_cache_put_get(self):
        """Test putting and getting embeddings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(temp_dir)

            code = "def hello(): print('hello')"
            model = "test-embedding-model"
            embedding = [0.1, 0.2, 0.3, 0.4]

            # Put embedding
            cache.put(code, model, embedding)

            # Get embedding
            retrieved = cache.get(code, model)
            assert retrieved == embedding

    def test_embedding_cache_miss(self):
        """Test embedding cache miss."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(temp_dir)

            result = cache.get("def test(): pass", "model")
            assert result is None

    def test_embedding_cache_clear(self):
        """Test clearing embedding cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(temp_dir)

            cache.put("def test(): pass", "model", [1, 2, 3])
            cache.clear()

            # Should be empty after clear
            result = cache.get("def test(): pass", "model")
            assert result is None


class TestCacheIntegration:
    """Integration tests for cache manager."""

    def test_cache_manager_integration(self):
        """Test integration between different cache types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(base_cache_dir=temp_dir)

            # Test explanation cache
            exp_cache = manager.get_explanation_cache()
            exp_cache.put("def add(a, b): return a + b", "vanilla", "codet5-base", "Adds two numbers")
            result = exp_cache.get("def add(a, b): return a + b", "vanilla", "codet5-base")
            assert result == "Adds two numbers"

            # Test advanced cache
            adv_cache = manager.get_advanced_cache()
            adv_cache.put("integration_key", "integration_value", tags={"test"})
            result = adv_cache.get("integration_key")
            assert result == "integration_value"

            # Test cache stats
            stats = manager.get_cache_stats()
            assert "explanation_cache" in stats
            assert "advanced_cache" in stats
            assert stats["advanced_cache"]["memory_entries"] == 1

    def test_cache_manager_clear_all(self):
        """Test clearing all caches through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(base_cache_dir=temp_dir)

            # Add data to caches
            exp_cache = manager.get_explanation_cache()
            exp_cache.put("def test(): pass", "vanilla", "model", "test explanation")

            adv_cache = manager.get_advanced_cache()
            adv_cache.put("test_key", "test_value")

            # Verify data exists
            assert exp_cache.size() == 1
            assert adv_cache.get("test_key") == "test_value"

            # Clear all
            manager.clear_all_caches()

            # Verify cleared
            assert exp_cache.size() == 0
            assert adv_cache.get("test_key") is None
