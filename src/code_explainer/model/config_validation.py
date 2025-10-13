"""Configuration validation methods for CodeExplainer class."""

import logging
from typing import Any, Dict
from ..exceptions import ConfigurationError


class CodeExplainerConfigValidationMixin:
    """Mixin class containing configuration validation methods for CodeExplainer."""

    def _validate_config(self, config: Any) -> None:
        """Validate configuration structure and values.

        Args:
            config: Configuration object to validate

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Use a safe logger before self.logger is available during early init
            _logger = getattr(self, "logger", logging.getLogger(__name__))
            cfg_dict = self._config_to_dict(config)
            if not cfg_dict:
                raise ConfigurationError("Configuration is empty or could not be converted to dict")

            # Validate critical configuration sections
            required_sections = ["model", "prompt"]
            for section in required_sections:
                if section not in cfg_dict:
                    _logger.warning(f"Configuration missing recommended section '{section}'")

            # Validate model configuration if present
            if "model" in cfg_dict:
                model_cfg = cfg_dict["model"]
                if not isinstance(model_cfg, dict):
                    raise ConfigurationError(f"Model configuration must be a dict, got {type(model_cfg)}")

                # Check for reasonable model parameters
                if "max_length" in model_cfg:
                    max_len = model_cfg["max_length"]
                    if not isinstance(max_len, int) or max_len <= 0 or max_len > 16384:
                        raise ConfigurationError(f"model.max_length must be positive integer <= 16384, got {max_len}")

                if "temperature" in model_cfg:
                    temp = model_cfg["temperature"]
                    if not isinstance(temp, (int, float)) or temp < 0 or temp > 2.0:
                        raise ConfigurationError(f"model.temperature must be float between 0 and 2.0, got {temp}")

            # Validate prompt configuration if present
            if "prompt" in cfg_dict:
                prompt_cfg = cfg_dict["prompt"]
                if not isinstance(prompt_cfg, dict):
                    raise ConfigurationError(f"Prompt configuration must be a dict, got {type(prompt_cfg)}")

                if "strategy" in prompt_cfg:
                    strategy = prompt_cfg["strategy"]
                    valid_strategies = ["vanilla", "ast_augmented", "multi_agent", "intelligent"]
                    if strategy not in valid_strategies:
                        _logger.warning(f"Prompt strategy '{strategy}' not in recommended strategies: {valid_strategies}")

            _logger.debug("Configuration validation passed")

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Configuration validation failed: {e}")

    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary.

        Args:
            config: Configuration object (OmegaConf, dict, etc.)

        Returns:
            Dictionary representation of configuration
        """
        return self._convert_omegaconf_to_dict(config)