"""
Configuration validation utilities.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import os


class ConfigValidator:
    """Validates configuration files and settings."""

    def __init__(self):
        self.required_fields = {
            'model': ['name'],
            'prompting': ['strategy']
        }
        self.valid_strategies = ['vanilla', 'ast_augmented', 'retrieval_augmented', 'execution_trace', 'enhanced_rag']
        self.errors = []
        self.warnings = []

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary."""
        self.errors = []
        self.warnings = []

        # Check required fields
        for section, fields in self.required_fields.items():
            if section not in config:
                self.errors.append(f"Required section '{section}' is missing")
                continue
            for field in fields:
                if field not in config[section]:
                    self.errors.append(f"Missing required field '{field}' in section '{section}'")

        # Validate specific values
        if 'prompting' in config and 'strategy' in config['prompting']:
            strategy = config['prompting']['strategy']
            if strategy not in self.valid_strategies:
                self.errors.append(f"Invalid prompting strategy: {strategy}")

        # Check for unknown sections (generate warnings)
        known_sections = set(self.required_fields.keys()) | {'model', 'training', 'data', 'cache', 'logging'}
        for section in config:
            if section not in known_sections:
                self.warnings.append(f"Unknown configuration section: '{section}'")

        return len(self.errors) == 0

    def validate_config_file(self, config_path: str) -> tuple[bool, List[str]]:
        """Validate configuration file."""
        path = Path(config_path)
        if not path.exists():
            return False, [f"Configuration file does not exist: {config_path}"]

        if not path.is_file():
            return False, [f"Configuration path is not a file: {config_path}"]

        # Basic file validation
        try:
            with open(path, 'r') as f:
                content = f.read()
                if not content.strip():
                    return False, ["Configuration file is empty"]
        except Exception as e:
            return False, [f"Error reading configuration file: {e}"]

        return True, []

    def get_config_template(self) -> Dict[str, Any]:
        """Get a configuration template."""
        return {
            'model': {
                'name': 'microsoft/CodeGPT-small-py',
                'arch': 'causal',
                'torch_dtype': 'auto',
                'load_in_8bit': False,
                'max_length': 512
            },
            'cache': {
                'enabled': True,
                'directory': '.cache',
                'max_size': 1000
            },
            'logging': {
                'level': 'INFO',
                'log_file': None
            },
            'training': {
                'output_dir': 'outputs',
                'num_train_epochs': 3,
                'per_device_train_batch_size': 4,
                'per_device_eval_batch_size': 4
            }
        }