"""Configuration initialization for Code Explainer."""

from pathlib import Path
from typing import Optional, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import Config, ModelConfig, TrainingConfig, CacheConfig, LoggingConfig, PromptConfig

# Register configs
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="model", name="default", node=ModelConfig)
cs.store(group="training", name="default", node=TrainingConfig)
cs.store(group="cache", name="default", node=CacheConfig)
cs.store(group="logging", name="default", node=LoggingConfig)
cs.store(group="prompt", name="default", node=PromptConfig)


def init_config(config_path: Optional[Union[str, Path]] = None) -> DictConfig:
    """Initialize configuration with Hydra.
    
    Args:
        config_path: Optional path to config file. If None, uses default config.
    
    Returns:
        DictConfig: Initialized configuration object
    """
    if config_path is None:
        # Use Hydra's compose API to load config with defaults
        with hydra.initialize_config_dir(
            str(Path(__file__).parent / "config")
        ):
            cfg = hydra.compose(config_name="config")
    else:
        # Load from specified YAML file
        cfg = OmegaConf.load(config_path)
        # Merge with defaults
        defaults = OmegaConf.load(Path(__file__).parent / "config/config.yaml")
        cfg = OmegaConf.merge(defaults, cfg)
    
    return cfg