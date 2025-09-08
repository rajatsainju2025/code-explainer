"""
Configuration management for evaluation system.

Handles YAML-based configurations with validation, inheritance,
and matrix experiment generation.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model inference."""
    name: str = "codet5-base"
    checkpoint: Optional[str] = None
    max_length: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    device: str = "auto"


@dataclass
class RetrievalConfig:
    """Configuration for RAG retrieval system."""
    enabled: bool = True
    top_k: int = 10
    similarity_threshold: float = 0.7
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    index_path: Optional[str] = None


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "bleu", "rouge_l", "latency", "cost"
    ])
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    significance_level: float = 0.05


@dataclass
class DatasetConfig:
    """Configuration for evaluation datasets."""
    train_path: Optional[str] = None
    eval_path: str = "data/eval.json"
    test_path: Optional[str] = None
    max_samples: Optional[int] = None
    shuffle: bool = True
    stratify: bool = False


@dataclass
class EvalConfig:
    """Main evaluation configuration."""
    name: str = "default_eval"
    description: str = ""
    
    # Core configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Experiment settings
    seed: int = 42
    output_dir: str = "results/eval"
    save_predictions: bool = True
    save_metrics: bool = True
    
    # Runtime settings
    batch_size: int = 1
    num_workers: int = 1
    verbose: bool = True


def load_config(config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> EvalConfig:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of configuration overrides
        
    Returns:
        EvalConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load base configuration
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Handle inheritance
    if 'extends' in config_data:
        parent_path = config_path.parent / config_data['extends']
        parent_config = load_config(parent_path)
        config_data = _merge_configs(parent_config.__dict__, config_data)
    
    # Apply overrides
    if overrides:
        config_data = _merge_configs(config_data, overrides)
    
    # Validate and create config object
    try:
        return _dict_to_config(config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def _dict_to_config(config_dict: Dict[str, Any]) -> EvalConfig:
    """Convert dictionary to EvalConfig with nested dataclasses."""
    # Extract nested configs
    model_config = ModelConfig(**config_dict.get('model', {}))
    retrieval_config = RetrievalConfig(**config_dict.get('retrieval', {}))
    metrics_config = MetricsConfig(**config_dict.get('metrics', {}))
    dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
    
    # Remove nested configs from main dict
    main_config = {k: v for k, v in config_dict.items() 
                   if k not in ['model', 'retrieval', 'metrics', 'dataset']}
    
    return EvalConfig(
        model=model_config,
        retrieval=retrieval_config,
        metrics=metrics_config,
        dataset=dataset_config,
        **main_config
    )


def save_config(config: EvalConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = _config_to_dict(config)
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def _config_to_dict(config: EvalConfig) -> Dict[str, Any]:
    """Convert EvalConfig to dictionary for serialization."""
    return {
        'name': config.name,
        'description': config.description,
        'model': config.model.__dict__,
        'retrieval': config.retrieval.__dict__,
        'metrics': config.metrics.__dict__,
        'dataset': config.dataset.__dict__,
        'seed': config.seed,
        'output_dir': config.output_dir,
        'save_predictions': config.save_predictions,
        'save_metrics': config.save_metrics,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'verbose': config.verbose
    }


def generate_matrix_configs(
    base_config: EvalConfig, 
    matrix: Dict[str, List[Any]],
    output_dir: Optional[str] = None
) -> List[EvalConfig]:
    """
    Generate multiple configurations from parameter matrix.
    
    Args:
        base_config: Base configuration to modify
        matrix: Dictionary mapping parameter paths to lists of values
        output_dir: Optional output directory for generated configs
        
    Returns:
        List of EvalConfig instances
        
    Example:
        matrix = {
            'model.temperature': [0.1, 0.5, 1.0],
            'retrieval.top_k': [5, 10, 20]
        }
    """
    import itertools
    
    # Generate all combinations
    param_names = list(matrix.keys())
    param_values = list(matrix.values())
    combinations = list(itertools.product(*param_values))
    
    configs = []
    for i, combo in enumerate(combinations):
        # Create copy of base config
        config_dict = _config_to_dict(base_config)
        
        # Apply parameter values
        for param_name, value in zip(param_names, combo):
            _set_nested_value(config_dict, param_name, value)
        
        # Update name and output directory
        config_dict['name'] = f"{base_config.name}_matrix_{i:03d}"
        if output_dir:
            config_dict['output_dir'] = f"{output_dir}/matrix_{i:03d}"
        
        # Convert back to EvalConfig
        config = _dict_to_config(config_dict)
        configs.append(config)
    
    return configs


def _set_nested_value(config_dict: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set nested dictionary value using dot notation."""
    keys = key_path.split('.')
    current = config_dict
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
