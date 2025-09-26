"""Hydra configuration for Code Explainer."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = MISSING
    arch: str = "causal"
    torch_dtype: str = "auto"
    load_in_8bit: bool = False
    max_length: int = 512
    device_map: Optional[str] = None
    cache_dir: Optional[str] = None
    device: str = "auto"  # auto, cuda, mps, cpu
    precision: str = "auto"  # auto, fp32, fp16, bf16, 8bit


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = MISSING
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    logging_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    warmup_steps: int = 0
    fp16: bool = False
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = True


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    directory: str = ".cache/explanations"
    max_size: int = 1000


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_file: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class PromptConfig:
    """Prompt configuration."""
    strategy: str = "vanilla"
    template: str = "Explain this Python code:\n```python\n{code}\n```\nExplanation:"


@dataclass
class Config:
    """Root configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)


defaults = [
    {"model": "default"},
    {"training": "default"},
    {"cache": "default"},
    {"logging": "default"},
    {"prompt": "default"},
    "_self_",
]