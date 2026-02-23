"""Configuration management utilities.

Optimized for:
- Cached configuration loading with larger cache
- Lazy YAML parser import
- Thread-safe logging setup
- Environment variable interpolation support
"""

import json
import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Lazy YAML import for faster startup
_yaml = None
_yaml_loader = None

def _get_yaml():
    """Lazily import yaml module and determine best loader."""
    global _yaml, _yaml_loader
    if _yaml is None:
        import yaml
        _yaml = yaml
        # Use C-based loader if available (5-10x faster)
        try:
            from yaml import CSafeLoader
            _yaml_loader = CSafeLoader
        except ImportError:
            _yaml_loader = yaml.SafeLoader
    return _yaml, _yaml_loader

# Pre-compiled regex for environment variable interpolation
_ENV_VAR_PATTERN = re.compile(r'\$\{([A-Z_][A-Z0-9_]*)\}')

# Thread safety for logging setup
_logging_lock = threading.Lock()
_logging_configured = False


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate environment variables in config values.
    
    Supports ${VAR_NAME} syntax for string values.
    """
    if isinstance(value, str):
        def replace_env(match):
            env_name = match.group(1)
            return os.environ.get(env_name, match.group(0))
        return _ENV_VAR_PATTERN.sub(replace_env, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


@lru_cache(maxsize=32)
def load_config(config_path: Union[str, Path], interpolate_env: bool = True) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file with caching.

    Args:
        config_path: Path to the configuration file
        interpolate_env: Whether to interpolate ${ENV_VAR} patterns

    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()
    
    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            yaml_mod, yaml_loader = _get_yaml()
            # Use fast C-based loader if available
            config = yaml_mod.load(f, Loader=yaml_loader)
        elif suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
    
    if interpolate_env and config:
        config = _interpolate_env_vars(config)
    
    return config or {}


def clear_config_cache() -> None:
    """Clear the configuration cache."""
    load_config.cache_clear()


def setup_logging(
    level: str = "INFO", 
    log_file: Optional[str] = None,
    force: bool = False
) -> None:
    """Setup logging configuration (thread-safe, idempotent).

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        force: Force reconfiguration even if already configured
    """
    global _logging_configured
    
    # Fast path: already configured
    if _logging_configured and not force:
        return
    
    with _logging_lock:
        if _logging_configured and not force:
            return
        
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # Clear existing handlers if forcing reconfiguration
        if force:
            root_logger.handlers.clear()
        
        # Set level
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # Add handlers if not present
        if not root_logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(console_handler)
            
            # File handler (optional)
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter(log_format))
                root_logger.addHandler(file_handler)
        
        _logging_configured = True