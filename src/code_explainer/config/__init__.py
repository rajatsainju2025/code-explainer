"""Configuration initialization for Code Explainer."""

from pathlib import Path
from typing import Optional, Union

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


def _validate_config_schema(cfg: DictConfig) -> None:
    """Validate config keys against expected schema to catch typos early.

    Args:
        cfg: Configuration to validate

    Raises:
        ValueError: If unknown keys are found in any config section
    """
    # Define expected keys for each section
    expected_keys = {
        "model": {
            "name", "arch", "torch_dtype", "load_in_8bit", "max_length",
            "temperature", "top_p", "top_k", "device_map", "cache_dir",
            "device", "precision"
        },
        "training": {
            "output_dir", "num_train_epochs", "per_device_train_batch_size",
            "per_device_eval_batch_size", "gradient_accumulation_steps",
            "learning_rate", "weight_decay", "logging_steps",
            "evaluation_strategy", "eval_steps", "save_steps",
            "save_total_limit", "load_best_model_at_end",
            "metric_for_best_model", "greater_is_better", "warmup_steps",
            "fp16", "bf16", "gradient_checkpointing", "torch_compile",
            "dataloader_num_workers", "remove_unused_columns"
        },
        "data": {
            "train_file", "eval_file", "test_file", "max_examples",
            "augment_ratio", "hub_id", "hub_split"
        },
        "cache": {"enabled", "directory", "max_size"},
        "logging": {"level", "log_file", "format", "date_format"},
        "prompt": {
            "strategy", "template", "language_templates",
            "max_code_length", "max_explanation_length"
        }
    }

    errors = []

    # Check each top-level section
    for section_name, expected in expected_keys.items():
        if section_name in cfg:
            section = cfg[section_name]
            if isinstance(section, DictConfig):
                actual_keys = set(section.keys())
                unknown_keys = actual_keys - expected
                if unknown_keys:
                    errors.append(
                        f"Unknown keys in '{section_name}' section: {sorted([str(k) for k in unknown_keys])}. "
                        f"Expected keys: {sorted(list(expected))}"
                    )

    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors) +
            "\n\nPlease check your config file for typos or remove unknown keys."
        )


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
        resolved = OmegaConf.create(OmegaConf.to_container(merged, resolve=True))
        if isinstance(resolved, DictConfig):
            _validate_config_schema(resolved)  # Validate schema
        return resolved

    # 2) Try repo default.yaml
    default_yaml = _find_default_yaml()
    if default_yaml is not None:
        defaults = OmegaConf.structured(Config())
        user = OmegaConf.load(str(default_yaml))
        merged = OmegaConf.merge(defaults, user)
        resolved = OmegaConf.create(OmegaConf.to_container(merged, resolve=True))
        if isinstance(resolved, DictConfig):
            _validate_config_schema(resolved)  # Validate schema
        return resolved  # type: ignore[return-value]

    # 3) Final fallback: structured defaults
    return OmegaConf.structured(Config())