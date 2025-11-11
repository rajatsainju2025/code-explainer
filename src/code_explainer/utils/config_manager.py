"""Configuration consolidation and centralized management.

Provides unified access to configuration from all sources:
- Hydra YAML configs
- Environment variables  
- Command-line arguments
- Runtime configuration

Classes:
    ConfigManager: Centralized configuration manager
"""

import os
from functools import lru_cache
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigManager:
    """Centralized configuration manager.
    
    Manages configuration from multiple sources with priority:
    1. Runtime config (highest priority)
    2. Environment variables
    3. Config file (Hydra YAML)
    4. Defaults (lowest priority)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Path to Hydra YAML config file
        """
        self.config_file = config_file
        self._config: Dict[str, Any] = {}
        self._env_overrides: Dict[str, Any] = {}
        self._runtime_config: Dict[str, Any] = {}
        # Cache for get() results
        self._get_cache: Dict[str, Any] = {}
    
    @lru_cache(maxsize=256)
    def get_env(self, key: str, default: str = "") -> str:
        """Get environment variable with caching.
        
        Args:
            key: Environment variable name
            default: Default value
        
        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback chain (optimized).
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        # Check runtime config first (most specific)
        if key in self._runtime_config:
            return self._runtime_config[key]
        
        # Check environment variables
        if key in self._env_overrides:
            return self._env_overrides[key]
        
        # Check loaded config
        if key in self._config:
            return self._config[key]
        
        # Return default
        return default
    
    def set(self, key: str, value: Any) -> None:
        """Set runtime configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._runtime_config[key] = value
    
    def update_from_env(self, prefix: str = "CODE_EXPLAINER_") -> None:
        """Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix to look for
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._env_overrides[config_key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary.
        
        Returns:
            Merged configuration dictionary
        """
        result = {}
        result.update(self._config)
        result.update(self._env_overrides)
        result.update(self._runtime_config)
        return result


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create global config manager.
    
    Returns:
        Global configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
