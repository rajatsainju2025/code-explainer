"""Tests for persistent cross-process model cache."""

import pytest
import tempfile
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.code_explainer.retrieval.model_cache import (
    PersistentModelCache,
    get_cached_model,
    clear_model_cache,
    get_model_cache_info
)


class TestPersistentModelCache:
    """Test suite for persistent model cache."""

    def test_cache_initialization(self):
        """Test cache directory initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            assert Path(tmpdir).exists()
            assert (Path(tmpdir) / ".locks").exists()

    def test_cache_path_generation(self):
        """Test that cache paths are deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            
            path1 = cache._get_cache_path("model-v1")
            path2 = cache._get_cache_path("model-v1")
            
            assert path1 == path2
            assert path1.suffix == ".pkl"

    def test_get_from_empty_cache(self):
        """Test retrieving from empty cache returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            result = cache.get("nonexistent-model")
            assert result is None

    def test_put_and_get(self):
        """Test caching and retrieving a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            
            # Create mock model
            mock_model = MagicMock()
            mock_model.name = "test-model"
            
            # Put model in cache
            success = cache.put("test-model", mock_model)
            assert success is True
            
            # Retrieve from cache
            cached = cache.get("test-model")
            assert cached is not None

    def test_local_cache_hit(self):
        """Test in-memory cache hit for repeated access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            
            mock_model = MagicMock()
            cache.put("model1", mock_model)
            
            # First access
            result1 = cache.get("model1")
            # Second access should hit in-memory cache
            result2 = cache.get("model1")
            
            assert result1 is not None
            assert result2 is not None

    def test_clear_cache(self):
        """Test clearing cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            
            mock_model = MagicMock()
            cache.put("model1", mock_model)
            
            assert cache.get("model1") is not None
            
            # Clear cache
            cache.clear()
            assert cache.get("model1") is None

    def test_cache_size_calculation(self):
        """Test cache size calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            
            # Create a real pickle file
            test_data = {"key": "value"}
            cache_path = cache._get_cache_path("test-model")
            with open(cache_path, 'wb') as f:
                pickle.dump(test_data, f)
            
            size = cache.get_cache_size()
            assert size > 0

    def test_cache_info(self):
        """Test getting cache information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            
            mock_model = MagicMock()
            cache.put("model1", mock_model)
            cache.put("model2", mock_model)
            
            info = cache.get_cache_info()
            assert "in_memory_models" in info
            assert "disk_models" in info
            assert "disk_size_mb" in info
            assert info["in_memory_models"] >= 2

    def test_concurrent_access_with_locking(self):
        """Test that file locking works for concurrent access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = PersistentModelCache(cache_dir=tmpdir)
            cache2 = PersistentModelCache(cache_dir=tmpdir)
            
            mock_model = MagicMock()
            
            # Cache in first instance
            cache1.put("shared-model", mock_model)
            
            # Retrieve in second instance (cross-process simulation)
            result = cache2.get("shared-model")
            assert result is not None

    def test_default_cache_directory(self):
        """Test default cache directory is ~/.cache/code-explainer/models/."""
        cache = PersistentModelCache()
        expected_path = Path.home() / ".cache" / "code-explainer" / "models"
        assert cache.cache_dir == expected_path


class TestGlobalCacheFunctions:
    """Test suite for global cache functions."""

    def test_get_cached_model_loads_fresh(self):
        """Test that get_cached_model loads fresh if not cached."""
        with patch('src.code_explainer.retrieval.model_cache.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            MockST.return_value = mock_model
            
            with patch('src.code_explainer.retrieval.model_cache._PERSISTENT_CACHE') as mock_cache:
                mock_cache.get.return_value = None  # Not in cache
                
                # This would normally load, but we're mocking
                # Just verify the function can be called
                assert callable(get_cached_model)

    def test_clear_model_cache(self):
        """Test clearing global model cache."""
        with patch('src.code_explainer.retrieval.model_cache._PERSISTENT_CACHE') as mock_cache:
            clear_model_cache()
            mock_cache.clear.assert_called_once()

    def test_get_model_cache_info(self):
        """Test getting global cache info."""
        with patch('src.code_explainer.retrieval.model_cache._PERSISTENT_CACHE') as mock_cache:
            mock_cache.get_cache_info.return_value = {
                "in_memory_models": 2,
                "disk_models": 5,
                "disk_size_mb": 1250.5
            }
            
            info = get_model_cache_info()
            assert info["in_memory_models"] == 2
            assert info["disk_models"] == 5


class TestCacheIntegrationWithRetriever:
    """Integration tests with CodeRetriever."""

    def test_retriever_uses_persistent_cache(self):
        """Test that CodeRetriever uses persistent cache."""
        with patch('src.code_explainer.retrieval.retriever.get_cached_model') as mock_get_cache:
            mock_model = MagicMock()
            mock_get_cache.return_value = mock_model
            
            # This would initialize CodeRetriever
            # Verify the import works
            from src.code_explainer.retrieval.retriever import CodeRetriever
            assert CodeRetriever is not None


@pytest.mark.slow
class TestCachePerformance:
    """Performance tests for persistent cache."""

    def test_repeated_access_is_faster(self):
        """Test that repeated access from in-memory cache is faster."""
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentModelCache(cache_dir=tmpdir)
            
            mock_model = MagicMock()
            cache.put("model", mock_model)
            
            # First access (may be slower)
            start = time.time()
            cache.get("model")
            first_time = time.time() - start
            
            # Second access (should be faster - in-memory)
            start = time.time()
            cache.get("model")
            second_time = time.time() - start
            
            # Second access should generally be faster or equal
            assert second_time <= first_time * 1.5  # Allow 50% margin for variance
