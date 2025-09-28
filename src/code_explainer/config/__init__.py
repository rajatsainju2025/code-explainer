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
    """Initialize configuration with robust fallbacks.

    Priority:
    1) Explicit YAML path when provided
    2) Repo default at '<repo_root>/configs/default.yaml' if available
    3) Structured defaults from dataclasses

    Returns a DictConfig in all cases.
    """
    # Helper to locate repo default.yaml reliably
    def _find_default_yaml() -> Optional[Path]:
        # src/code_explainer/config/__init__.py -> parents[3] should be repo root
        try:
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / "configs" / "default.yaml"
            if candidate.exists():
                return candidate
        except Exception:
            pass
        # Fallback: try one level up (in case of unusual layouts)
        try:
            alt_root = Path(__file__).resolve().parents[2]
            candidate = alt_root / "configs" / "default.yaml"
            if candidate.exists():
                return candidate
        except Exception:
            pass
        return None

    # 1) Explicit config path
    if config_path is not None:
        cfg = OmegaConf.load(str(config_path))
        # Merge with structured defaults to ensure all keys exist
        defaults = OmegaConf.structured(Config())
        merged = OmegaConf.merge(defaults, cfg)
        return OmegaConf.create(OmegaConf.to_container(merged, resolve=True))  # type: ignore[return-value]

    # 2) Try repo default.yaml
    default_yaml = _find_default_yaml()
    if default_yaml is not None:
        defaults = OmegaConf.structured(Config())
        user = OmegaConf.load(str(default_yaml))
        merged = OmegaConf.merge(defaults, user)
        return OmegaConf.create(OmegaConf.to_container(merged, resolve=True))  # type: ignore[return-value]

    # 3) Final fallback: structured defaults
    return OmegaConf.structured(Config())