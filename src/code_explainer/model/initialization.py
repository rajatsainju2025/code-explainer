"""Initialization methods for CodeExplainer class."""

from pathlib import Path
from typing import Any, Optional, Union, Tuple
import logging

from ..cache import ExplanationCache
from ..config import Config, init_config
from ..error_handling import setup_logging
from ..model_loader import ModelLoader, ModelResources
from ..exceptions import ConfigurationError


class CodeExplainerInitializationMixin:
    """Mixin class containing initialization methods for CodeExplainer."""

    def _initialize_config(
        self,
        model_path: Optional[Union[str, Path, Any]],
        config_path: Optional[str]
    ) -> Tuple[Any, Optional[Union[str, Path]]]:
        """Initialize configuration from various sources.

        Args:
            model_path: Model path or config object
            config_path: Path to config file

        Returns:
            Tuple of (config, resolved_model_path)
        """
        # Determine if first argument is a config object or a model path
        user_provided_config = None
        resolved_model_path = model_path

        if model_path is not None and not isinstance(model_path, (str, Path)):
            # Treat as config-like object
            user_provided_config = model_path
            resolved_model_path = None

        # Initialize configuration
        try:
            if user_provided_config is not None:
                config = user_provided_config  # type: ignore
            else:
                config = init_config(config_path)

            # Validate the configuration
            self._validate_config(config)

            return config, resolved_model_path

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {e}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the code explainer.

        Returns:
            Configured logger instance
        """
        try:
            # Extract logging settings from config
            cfg_dict = self._config_to_dict(self.config)
            level, log_file = self._get_logging_settings(cfg_dict)

            # Setup logging with extracted settings
            setup_logging(level=level, log_file=log_file)

            # Get logger for this module
            logger = logging.getLogger(__name__)
            logger.debug("Logging initialized successfully")
            return logger

        except Exception as e:
            # Fallback to basic logging setup
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to setup advanced logging, using basic setup: {e}")
            return logger

    def _initialize_model_resources(self, model_path: Optional[Union[str, Path]]) -> Optional[ModelResources]:
        """Initialize model resources (model, tokenizer, etc.).

        Args:
            model_path: Path to the model directory

        Returns:
            ModelResources instance or None if initialization fails
        """
        try:
            if model_path is None:
                self.logger.debug("No model path provided, deferring model loading until first access")
                return None

            self.logger.debug(f"Loading model resources from: {model_path}")

            # Initialize model loader
            self.model_loader = ModelLoader(self.config.model)

            # Load model resources
            resources = self.model_loader.load(model_path)

            self.logger.debug("Model resources initialized successfully")
            return resources

        except Exception as e:
            self.logger.error(f"Failed to initialize model resources: {e}")
            return None

    def _initialize_components(self) -> None:
        """Initialize additional components like cache, etc."""
        try:
            # Initialize explanation cache if configured
            cache_enabled = self._cfg_get_bool("cache.enabled", False)
            if cache_enabled:
                cache_dir = self._cfg_get_str("cache.dir", "./cache")
                cache_size = self._cfg_get_int("cache.max_size", 1000)

                self.explanation_cache = ExplanationCache(
                    cache_dir=cache_dir,
                    max_size=cache_size
                )
                self.logger.info(f"Explanation cache initialized with max size: {cache_size}")
            else:
                self.explanation_cache = None
                self.logger.debug("Explanation cache disabled")

            # Initialize other components as needed
            self.cache_manager = None
            self.advanced_cache = None

        except Exception as e:
            self.logger.warning(f"Failed to initialize some components: {e}")
            self.explanation_cache = None
            self.cache_manager = None
            self.advanced_cache = None