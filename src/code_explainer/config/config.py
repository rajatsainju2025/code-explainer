"""Hydra configuration for Code Explainer."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = MISSING
    arch: str = "causal"
    torch_dtype: str = "auto"
    load_in_8bit: bool = False
    max_length: int = 512
    # Sampling/generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
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
    bf16: bool = False
    gradient_checkpointing: bool = False
    torch_compile: bool = False
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
    # Optional per-language templates to override the default template
    language_templates: Dict[str, str] = field(default_factory=dict)
    # Soft limits for prompt/explanation sizes
    max_code_length: int = 200
    max_explanation_length: int = 300


@dataclass
class DataConfig:
    """Data configuration for training/eval files and dataset options."""
    train_file: Optional[str] = "data/train.json"
    eval_file: Optional[str] = "data/eval.json"
    test_file: Optional[str] = "data/test.json"
    max_examples: Optional[int] = None
    augment_ratio: float = 0.0
    hub_id: Optional[str] = None
    hub_split: str = "train"


@dataclass
class Config:
    """Root configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
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