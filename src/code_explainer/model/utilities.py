"""Utility methods for CodeExplainer class."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
from omegaconf import OmegaConf, DictConfig, ListConfig


class CodeExplainerUtilitiesMixin:
    """Mixin class containing utility methods for CodeExplainer."""

    def _cfg_get_typed(self, dotted_path: str, expected_type: type, default: Any) -> Any:
        """Safely get a configuration value with type checking.

        Args:
            dotted_path: Configuration path (e.g., "model.max_length")
            expected_type: Expected type of the value
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        try:
            value = self._safe_config_access(self.config, dotted_path)
            if value is not None and isinstance(value, expected_type):
                return value
        except Exception:
            pass
        return default

    def _cfg_get_int(self, dotted_path: str, default: int = 0) -> int:
        """Get integer configuration value with validation.

        Args:
            dotted_path: Configuration path
            default: Default value

        Returns:
            Validated integer value
        """
        return self._cfg_get_typed(dotted_path, int, default)

    def _cfg_get_float(self, dotted_path: str, default: float = 0.0) -> float:
        """Get float configuration value with validation.

        Args:
            dotted_path: Configuration path
            default: Default value

        Returns:
            Validated float value
        """
        return self._cfg_get_typed(dotted_path, float, default)

    def _cfg_get_str(
        self,
        dotted_path: str,
        default: str = "",
        valid_values: Optional[list] = None
    ) -> str:
        """Get string configuration value with validation.

        Args:
            dotted_path: Configuration path
            default: Default value
            valid_values: List of valid string values

        Returns:
            Validated string value
        """
        value = self._cfg_get_typed(dotted_path, str, default)

        if valid_values is not None and value not in valid_values:
            self.logger.warning(f"Config '{dotted_path}' value '{value}' not in valid values {valid_values}, using default: {default}")
            return default

        return value

    def _cfg_get_bool(self, dotted_path: str, default: bool = False) -> bool:
        """Get boolean configuration value with validation.

        Args:
            dotted_path: Configuration path
            default: Default value

        Returns:
            Boolean value
        """
        return self._cfg_get_typed(dotted_path, bool, default)

    def _get_logging_settings(self, cfg_dict: Dict[str, Any]) -> Tuple[str, Optional[Union[str, Path]]]:
        """Extract safe logging settings from a config dict.

        Ensures the log level is a valid string level and log_file is a str/Path if provided.
        """
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        default_level = "INFO"
        level: str = default_level
        log_file: Optional[Union[str, Path]] = None

        if isinstance(cfg_dict, dict):
            logging_cfg = cfg_dict.get("logging")
            if isinstance(logging_cfg, dict):
                lvl = logging_cfg.get("level", default_level)
                if isinstance(lvl, str) and lvl.upper() in valid_levels:
                    level = lvl
                elif isinstance(lvl, int):
                    # Convert numeric to name if possible
                    try:
                        name = logging.getLevelName(lvl)
                        if isinstance(name, str) and name.upper() in valid_levels:
                            level = name
                    except Exception:
                        pass

                lf = logging_cfg.get("log_file")
                if isinstance(lf, (str, Path)):
                    log_file = lf

        return level, log_file

    def _safe_config_access(self, config: Any, dotted_path: str) -> Any:
        """Safely access nested configuration values.

        Args:
            config: Configuration object
            dotted_path: Dotted path to access (e.g., "model.max_length")

        Returns:
            Configuration value or None if not found
        """
        try:
            keys = dotted_path.split('.')
            current = config

            for key in keys:
                if hasattr(current, key):
                    current = getattr(current, key)
                elif isinstance(current, dict):
                    current = current.get(key)
                elif hasattr(current, '__getitem__'):
                    current = current[key]
                else:
                    return None

            return current
        except Exception:
            return None

    def _convert_omegaconf_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert OmegaConf configuration to plain dictionary.

        Args:
            config: OmegaConf or dict configuration

        Returns:
            Plain dictionary representation
        """
        if isinstance(config, (DictConfig, ListConfig)):
            return OmegaConf.to_container(config, resolve=True)  # type: ignore
        elif isinstance(config, dict):
            return config
        else:
            # Try to convert to dict if possible
            try:
                return dict(config)
            except Exception:
                return {}