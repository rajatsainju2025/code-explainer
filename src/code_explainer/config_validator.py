"""
Configuration validation utilities.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os

# Pre-computed constant sets for fast lookups
REQUIRED_FIELDS = {
    'model': frozenset(['name']),
    'prompting': frozenset(['strategy'])
}
VALID_STRATEGIES = frozenset(['vanilla', 'ast_augmented', 'retrieval_augmented', 'execution_trace', 'enhanced_rag'])
KNOWN_SECTIONS = frozenset(['model', 'prompting', 'training', 'data', 'cache', 'logging'])


class ConfigValidator:
    """Validates configuration files and settings."""

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary with O(1) lookups."""
        self.errors = []
        self.warnings = []

        # Check required fields efficiently
        for section, fields in REQUIRED_FIELDS.items():
            if section not in config:
                self.errors.append(f"Required section '{section}' is missing")
                continue
            section_config = config[section]
            for field in fields:
                if field not in section_config:
                    self.errors.append(f"Missing required field '{field}' in section '{section}'")

        # Validate specific values with O(1) set lookup
        if 'prompting' in config and 'strategy' in config['prompting']:
            strategy = config['prompting']['strategy']
            if strategy not in VALID_STRATEGIES:
                self.errors.append(f"Invalid prompting strategy: {strategy}")

        # Check for unknown sections efficiently
        for section in config:
            if section not in KNOWN_SECTIONS:
                self.warnings.append(f"Unknown configuration section: '{section}'")

        return len(self.errors) == 0

    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str]]:
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