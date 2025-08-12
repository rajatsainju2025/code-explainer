"""Configuration validation utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates code explainer configuration files."""

    REQUIRED_SECTIONS = ["model", "prompting"]

    SECTION_SCHEMAS = {
        "model": {
            "required": ["name"],
            "optional": ["cache_dir", "torch_dtype", "load_in_8bit", "device_map", "max_length"],
        },
        "training": {
            "required": ["output_dir"],
            "optional": [
                "num_train_epochs",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "weight_decay",
                "logging_steps",
                "evaluation_strategy",
                "eval_steps",
                "save_steps",
                "save_total_limit",
                "load_best_model_at_end",
                "metric_for_best_model",
                "greater_is_better",
                "warmup_steps",
                "fp16",
                "dataloader_num_workers",
                "remove_unused_columns",
            ],
        },
        "data": {
            "required": [],
            "optional": ["train_file", "eval_file", "test_file", "max_samples"],
        },
        "generation": {
            "required": [],
            "optional": [
                "max_new_tokens",
                "num_beams",
                "do_sample",
                "temperature",
                "top_p",
                "pad_token_id",
            ],
        },
        "prompting": {
            "required": ["strategy"],
            "optional": ["max_context_length", "include_docstrings", "include_imports"],
        },
        "retrieval": {
            "required": [],
            "optional": [
                "index_path",
                "embedding_model",
                "index_dimension",
                "similarity_top_k",
                "similarity_threshold",
                "chunk_size",
                "chunk_overlap",
            ],
        },
        "multi_agent": {"required": [], "optional": ["enabled", "agents"]},
        "symbolic": {
            "required": [],
            "optional": [
                "enabled",
                "max_execution_time",
                "include_complexity",
                "include_dataflow",
                "generate_tests",
            ],
        },
        "evaluation": {"required": [], "optional": ["metrics", "batch_size", "max_samples"]},
        "logging": {"required": [], "optional": ["level", "format", "file"]},
        "web": {"required": [], "optional": ["gradio", "api"]},
    }

    VALID_STRATEGIES = [
        "vanilla",
        "ast_augmented",
        "retrieval_augmented",
        "execution_trace",
        "enhanced_rag",
    ]

    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def __init__(self):
        """Initialize the configuration validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_config(self, config: Union[Dict[str, Any], str, Path]) -> bool:
        """Validate a configuration dictionary or file.

        Args:
            config: Configuration dictionary or path to config file

        Returns:
            True if valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        # Load config if it's a file path
        if isinstance(config, (str, Path)):
            try:
                config = self._load_config_file(config)
            except Exception as e:
                self.errors.append(f"Failed to load config file: {e}")
                return False

        # Validate structure
        self._validate_structure(config)

        # Validate specific sections
        for section_name, section_config in config.items():
            self._validate_section(section_name, section_config)

        # Log results
        if self.errors:
            logger.error(f"Configuration validation failed with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning(f"Configuration has {len(self.warnings)} warnings")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        return len(self.errors) == 0

    def _load_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _validate_structure(self, config: Dict[str, Any]) -> None:
        """Validate overall configuration structure."""
        if not isinstance(config, dict):
            self.errors.append("Configuration must be a dictionary")
            return

        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                self.errors.append(f"Required section '{section}' is missing")

        # Check for unknown sections
        known_sections = set(self.SECTION_SCHEMAS.keys())
        for section in config.keys():
            if section not in known_sections:
                self.warnings.append(f"Unknown configuration section: '{section}'")

    def _validate_section(self, section_name: str, section_config: Any) -> None:
        """Validate a specific configuration section."""
        if section_name not in self.SECTION_SCHEMAS:
            return  # Already warned about unknown sections

        schema = self.SECTION_SCHEMAS[section_name]

        if not isinstance(section_config, dict):
            self.errors.append(f"Section '{section_name}' must be a dictionary")
            return

        # Check required fields
        for required_field in schema["required"]:
            if required_field not in section_config:
                self.errors.append(
                    f"Required field '{required_field}' missing in section '{section_name}'"
                )

        # Check for unknown fields
        known_fields = set(schema["required"] + schema["optional"])
        for field in section_config.keys():
            if field not in known_fields:
                self.warnings.append(f"Unknown field '{field}' in section '{section_name}'")

        # Validate specific fields
        self._validate_section_fields(section_name, section_config)

    def _validate_section_fields(self, section_name: str, section_config: Dict[str, Any]) -> None:
        """Validate specific fields within a section."""

        # Validate prompting strategy
        if section_name == "prompting" and "strategy" in section_config:
            strategy = section_config["strategy"]
            if strategy not in self.VALID_STRATEGIES:
                self.errors.append(
                    f"Invalid prompting strategy '{strategy}'. "
                    f"Valid options: {', '.join(self.VALID_STRATEGIES)}"
                )

        # Validate logging level
        if section_name == "logging" and "level" in section_config:
            level = section_config["level"].upper()
            if level not in self.VALID_LOG_LEVELS:
                self.errors.append(
                    f"Invalid logging level '{level}'. "
                    f"Valid options: {', '.join(self.VALID_LOG_LEVELS)}"
                )

        # Validate file paths
        if section_name == "data":
            for file_key in ["train_file", "eval_file", "test_file"]:
                if file_key in section_config:
                    file_path = section_config[file_key]
                    if file_path and not Path(file_path).exists():
                        self.warnings.append(f"Data file does not exist: {file_path}")

        # Validate retrieval configuration
        if section_name == "retrieval":
            if "index_path" in section_config:
                index_path = section_config["index_path"]
                if index_path and not Path(index_path).exists():
                    self.warnings.append(f"FAISS index file does not exist: {index_path}")

            if "similarity_top_k" in section_config:
                k = section_config["similarity_top_k"]
                if not isinstance(k, int) or k <= 0:
                    self.errors.append("similarity_top_k must be a positive integer")

            if "similarity_threshold" in section_config:
                threshold = section_config["similarity_threshold"]
                if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                    self.errors.append("similarity_threshold must be a number between 0 and 1")

        # Validate training configuration
        if section_name == "training":
            if "num_train_epochs" in section_config:
                epochs = section_config["num_train_epochs"]
                if not isinstance(epochs, (int, float)) or epochs <= 0:
                    self.errors.append("num_train_epochs must be a positive number")

            if "learning_rate" in section_config:
                lr = section_config["learning_rate"]
                if not isinstance(lr, (int, float)) or lr <= 0:
                    self.errors.append("learning_rate must be a positive number")

        # Validate model configuration
        if section_name == "model":
            if "max_length" in section_config:
                max_len = section_config["max_length"]
                if not isinstance(max_len, int) or max_len <= 0:
                    self.errors.append("max_length must be a positive integer")

    def get_validation_report(self) -> str:
        """Get a formatted validation report."""
        report = []

        if self.errors:
            report.append("❌ ERRORS:")
            for error in self.errors:
                report.append(f"  - {error}")

        if self.warnings:
            if report:
                report.append("")
            report.append("⚠️  WARNINGS:")
            for warning in self.warnings:
                report.append(f"  - {warning}")

        if not self.errors and not self.warnings:
            report.append("✅ Configuration is valid")

        return "\n".join(report)


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """Convenience function to validate a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator()
    is_valid = validator.validate_config(config_path)

    print(validator.get_validation_report())
    return is_valid


def main():
    """CLI entry point for config validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate code explainer configuration files")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    is_valid = validate_config_file(args.config)
    exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
