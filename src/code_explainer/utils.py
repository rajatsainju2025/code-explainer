"""Configuration management utilities."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def get_device() -> str:
    """Get the best available device for training/inference."""
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
