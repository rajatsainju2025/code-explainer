"""Tests for lazy model loading functionality."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.code_explainer.model.core import CodeExplainer


class TestLazyModelLoading:
    """Test suite for lazy model loading behavior."""

    def test_model_not_loaded_on_init(self, tmp_path):
        """Verify that model resources are not loaded during initialization."""
        config_path = "configs/default.yaml"
        
        with patch('src.code_explainer.model.core.ModelLoader') as mock_loader:
            # Initialize CodeExplainer without loading model
            explainer = CodeExplainer(
                model_path=str(tmp_path / "dummy_model"),
                config_path=config_path
            )
            
            # Model loader should not have been called during __init__
            mock_loader.assert_not_called()
            
            # Verify resources are not yet loaded
            assert explainer._resources is None
            assert explainer.is_model_loaded is False

    def test_model_loaded_on_first_access(self, tmp_path):
        """Verify that model is loaded on first property access."""
        config_path = "configs/default.yaml"
        model_path = str(tmp_path / "dummy_model")
        
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_device = MagicMock()
        
        with patch('src.code_explainer.model.core.ModelLoader') as MockModelLoader:
            # Setup mock loader
            mock_loader_instance = MagicMock()
            MockModelLoader.return_value = mock_loader_instance
            
            # Mock the load method to return resources
            from src.code_explainer.model_loader import ModelResources
            mock_resources = ModelResources(
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=mock_device,
                model_type="causal"
            )
            mock_loader_instance.load.return_value = mock_resources
            
            # Initialize CodeExplainer
            explainer = CodeExplainer(
                model_path=model_path,
                config_path=config_path
            )
            
            # First access should trigger loading
            assert explainer.is_model_loaded is False
            _ = explainer.model  # Access model property
            
            # Now resources should be loaded
            assert explainer.is_model_loaded is True
            mock_loader_instance.load.assert_called_once()

    def test_tokenizer_access_triggers_lazy_loading(self, tmp_path):
        """Verify that accessing tokenizer also triggers lazy loading."""
        config_path = "configs/default.yaml"
        model_path = str(tmp_path / "dummy_model")
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_device = MagicMock()
        
        with patch('src.code_explainer.model.core.ModelLoader') as MockModelLoader:
            mock_loader_instance = MagicMock()
            MockModelLoader.return_value = mock_loader_instance
            
            from src.code_explainer.model_loader import ModelResources
            mock_resources = ModelResources(
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=mock_device,
                model_type="causal"
            )
            mock_loader_instance.load.return_value = mock_resources
            
            explainer = CodeExplainer(
                model_path=model_path,
                config_path=config_path
            )
            
            # Access tokenizer instead of model
            _ = explainer.tokenizer
            
            # Resources should be loaded
            assert explainer.is_model_loaded is True
            mock_loader_instance.load.assert_called_once()

    def test_model_loaded_only_once(self, tmp_path):
        """Verify that model is loaded only once even with multiple accesses."""
        config_path = "configs/default.yaml"
        model_path = str(tmp_path / "dummy_model")
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_device = MagicMock()
        
        with patch('src.code_explainer.model.core.ModelLoader') as MockModelLoader:
            mock_loader_instance = MagicMock()
            MockModelLoader.return_value = mock_loader_instance
            
            from src.code_explainer.model_loader import ModelResources
            mock_resources = ModelResources(
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=mock_device,
                model_type="causal"
            )
            mock_loader_instance.load.return_value = mock_resources
            
            explainer = CodeExplainer(
                model_path=model_path,
                config_path=config_path
            )
            
            # Access model multiple times
            _ = explainer.model
            _ = explainer.tokenizer
            _ = explainer.model
            
            # Load should have been called only once
            mock_loader_instance.load.assert_called_once()

    def test_injected_model_bypasses_lazy_loading(self, tmp_path):
        """Verify that injected models don't trigger lazy loading."""
        config_path = "configs/default.yaml"
        
        mock_model = MagicMock()
        
        with patch('src.code_explainer.model.core.ModelLoader') as MockModelLoader:
            explainer = CodeExplainer(
                model_path=str(tmp_path / "dummy_model"),
                config_path=config_path
            )
            
            # Inject a model
            explainer.model = mock_model
            
            # Accessing model should return injected model without loading
            assert explainer.model == mock_model
            MockModelLoader.assert_not_called()

    def test_no_model_path_defers_loading_gracefully(self, tmp_path):
        """Verify that missing model path doesn't break lazy loading."""
        config_path = "configs/default.yaml"
        
        with patch('src.code_explainer.model.core.ModelLoader'):
            explainer = CodeExplainer(
                model_path=None,
                config_path=config_path
            )
            
            # Should not be loaded
            assert explainer.is_model_loaded is False
            assert explainer._model_path is None

    def test_is_model_loaded_property(self, tmp_path):
        """Test is_model_loaded property accuracy."""
        config_path = "configs/default.yaml"
        
        with patch('src.code_explainer.model.core.ModelLoader'):
            explainer = CodeExplainer(
                model_path=str(tmp_path / "dummy_model"),
                config_path=config_path
            )
            
            # Before loading
            assert explainer.is_model_loaded is False
            
            # After injection
            explainer.model = MagicMock()
            assert explainer.is_model_loaded is False  # Still false because _resources is None
            
            # After actual loading
            from src.code_explainer.model_loader import ModelResources
            explainer._resources = ModelResources(
                model=MagicMock(),
                tokenizer=MagicMock(),
                device=MagicMock(),
                model_type="causal"
            )
            assert explainer.is_model_loaded is True


@pytest.mark.slow
class TestLazyLoadingPerformance:
    """Performance tests for lazy loading."""

    def test_init_faster_than_model_loading(self, tmp_path):
        """Verify that init is significantly faster with lazy loading."""
        config_path = "configs/default.yaml"
        
        with patch('src.code_explainer.model.core.ModelLoader'):
            # Time initialization (should be fast without model loading)
            start = time.time()
            explainer = CodeExplainer(
                model_path=str(tmp_path / "dummy_model"),
                config_path=config_path
            )
            init_time = time.time() - start
            
            # Should complete in < 1 second (no model loading)
            # This is a loose bound; adjust based on actual system performance
            assert init_time < 2.0, f"Initialization took too long: {init_time}s"
