"""Fast-path validators to eliminate overhead from complex validation logic.

This module provides pre-compiled and optimized validators for common patterns
to avoid repeated validation computations in hot paths.
"""

import re
from typing import Any, Callable, Pattern, Optional, Set, List
import threading


class FastPathValidator:
    """Optimized validators with pre-compiled patterns and early-exit logic."""
    
    __slots__ = ('_patterns', '_validators', '_lock')
    
    def __init__(self):
        """Initialize the fast-path validator."""
        # Pre-compile commonly used regex patterns
        self._patterns: dict[str, Pattern] = {
            'code_identifier': re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
            'strategy_name': re.compile(r'^[a-z_]+$'),
            'model_name': re.compile(r'^[\w\-\.]+/[\w\-\.]+$'),
            'numeric_identifier': re.compile(r'^\d+$'),
            'url_safe': re.compile(r'^[a-zA-Z0-9\-_.~]*$'),
        }
        
        # Cache validators
        self._validators: dict[str, Callable] = {}
        self._lock = threading.RLock()
    
    def validate_code_identifier(self, value: str) -> bool:
        """Fast validation for code identifiers (variable/function names)."""
        if not isinstance(value, str):
            return False
        if not value:  # Empty string short-circuit
            return False
        if len(value) > 255:  # Identifier length limit
            return False
        return bool(self._patterns['code_identifier'].match(value))
    
    def validate_strategy(self, value: str, allowed: Optional[Set[str]] = None) -> bool:
        """Fast validation for strategy names."""
        if not isinstance(value, str):
            return False
        if not value:
            return False
        
        # If allowed set provided, use fast set lookup
        if allowed is not None:
            return value in allowed
        
        # Otherwise use pattern
        return bool(self._patterns['strategy_name'].match(value))
    
    def validate_non_empty_string(self, value: Any, max_length: int = 65536) -> bool:
        """Fast validation for non-empty strings with max length."""
        if not isinstance(value, str):
            return False
        return 0 < len(value) <= max_length
    
    def validate_positive_int(self, value: Any, max_value: Optional[int] = None) -> bool:
        """Fast validation for positive integers."""
        if not isinstance(value, int):
            return False
        if value <= 0:
            return False
        if max_value is not None and value > max_value:
            return False
        return True
    
    def validate_model_name(self, value: str) -> bool:
        """Fast validation for model names (e.g., 'org/model-name')."""
        if not isinstance(value, str):
            return False
        if not value:
            return False
        return bool(self._patterns['model_name'].match(value))
    
    def validate_list_of_strings(self, value: Any, min_items: int = 1, 
                                max_items: Optional[int] = None) -> bool:
        """Fast validation for list of strings."""
        if not isinstance(value, list):
            return False
        if len(value) < min_items:
            return False
        if max_items is not None and len(value) > max_items:
            return False
        # Check all items are strings (early exit on first non-string)
        return all(isinstance(item, str) and item for item in value)
    
    def validate_dict_structure(self, value: Any, required_keys: Optional[Set[str]] = None,
                               key_types: Optional[dict[str, type]] = None) -> bool:
        """Fast validation for dict structure."""
        if not isinstance(value, dict):
            return False
        
        # Check required keys
        if required_keys and not required_keys.issubset(value.keys()):
            return False
        
        # Check key types
        if key_types:
            for key, expected_type in key_types.items():
                if key in value and not isinstance(value[key], expected_type):
                    return False
        
        return True


# Pre-compiled patterns for import validators
_IMPORT_PATTERNS = {
    'relative_import': re.compile(r'^\.+\w+'),
    'absolute_import': re.compile(r'^[a-zA-Z_]\w*'),
    'dotted_import': re.compile(r'^[\w.]+$'),
}


class ImportValidator:
    """Optimized validators for Python import statements."""
    
    __slots__ = ('_patterns',)
    
    def __init__(self):
        """Initialize import validator."""
        self._patterns = _IMPORT_PATTERNS
    
    def is_valid_relative_import(self, path: str) -> bool:
        """Check if string is valid relative import."""
        if not isinstance(path, str) or not path:
            return False
        return bool(self._patterns['relative_import'].match(path))
    
    def is_valid_absolute_import(self, path: str) -> bool:
        """Check if string is valid absolute import."""
        if not isinstance(path, str) or not path:
            return False
        return bool(self._patterns['absolute_import'].match(path))
    
    def is_valid_dotted_import(self, path: str) -> bool:
        """Check if string is valid dotted import path."""
        if not isinstance(path, str) or not path:
            return False
        return bool(self._patterns['dotted_import'].match(path))


class CachedSetValidator:
    """Validator that caches set membership for repeated validations."""
    
    __slots__ = ('_values', '_frozen_set', '_lock')
    
    def __init__(self, values: List[str]):
        """Initialize with list of allowed values.
        
        Args:
            values: List of allowed values to create cached set
        """
        self._values = values
        self._frozen_set = frozenset(values)  # Immutable for thread-safety
        self._lock = threading.RLock()
    
    def is_valid(self, value: Any) -> bool:
        """O(1) check if value is in allowed set."""
        return value in self._frozen_set
    
    def validate_all(self, values: List[Any]) -> bool:
        """Check if all values are in allowed set."""
        return all(v in self._frozen_set for v in values)
    
    def get_invalid(self, values: List[Any]) -> List[Any]:
        """Get list of values not in allowed set."""
        return [v for v in values if v not in self._frozen_set]


# Global instances
_fast_validator = FastPathValidator()
_import_validator = ImportValidator()


def get_fast_validator() -> FastPathValidator:
    """Get singleton FastPathValidator instance."""
    return _fast_validator


def get_import_validator() -> ImportValidator:
    """Get singleton ImportValidator instance."""
    return _import_validator


def create_strategy_validator(allowed_strategies: List[str]) -> CachedSetValidator:
    """Create optimized validator for strategy names."""
    return CachedSetValidator(allowed_strategies)


# Pre-built validators for common use cases
STRATEGY_VALIDATOR_VANILLA = CachedSetValidator(['vanilla', 'detailed', 'concise'])
LANGUAGE_VALIDATOR = CachedSetValidator(['python', 'java', 'cpp', 'javascript', 'go', 'rust', 'c'])
