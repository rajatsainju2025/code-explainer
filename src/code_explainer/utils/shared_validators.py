"""Shared validation utilities to eliminate code duplication."""

from typing import Optional, List, Any
import re


class ValidationUtils:
    """Common validation functions used across modules."""

    # Pre-compiled patterns for efficiency
    CODE_PATTERN = re.compile(r'^[\s\S]*$')
    LANGUAGE_PATTERN = re.compile(r'^[a-z\-]+$')
    IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    NUMERIC_PATTERN = re.compile(r'^\d+(\.\d+)?$')

    @staticmethod
    def validate_code_input(code: str, max_length: int = 100000) -> tuple[bool, Optional[str]]:
        """
        Validate code input string.
        
        Args:
            code: Code to validate
            max_length: Maximum allowed code length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(code, str):
            return False, "Code must be a string"
        
        if not code or not code.strip():
            return False, "Code cannot be empty"
        
        if len(code) > max_length:
            return False, f"Code exceeds maximum length of {max_length} characters"
        
        return True, None

    @staticmethod
    def validate_strategy(strategy: Optional[str], allowed: List[str]) -> tuple[bool, Optional[str]]:
        """Validate explanation strategy."""
        if strategy is None:
            return True, None
        
        if not isinstance(strategy, str):
            return False, "Strategy must be a string"
        
        if strategy not in allowed:
            return False, f"Strategy must be one of: {', '.join(allowed)}"
        
        return True, None

    @staticmethod
    def validate_numeric_range(value: Any, min_val: float = 0.0, max_val: float = 1.0,
                              name: str = "value") -> tuple[bool, Optional[str]]:
        """Validate numeric value is within range."""
        try:
            num_value = float(value)
        except (TypeError, ValueError):
            return False, f"{name} must be numeric"
        
        if num_value < min_val or num_value > max_val:
            return False, f"{name} must be between {min_val} and {max_val}"
        
        return True, None

    @staticmethod
    def validate_integer_range(value: Any, min_val: int = 0, max_val: Optional[int] = None,
                               name: str = "value") -> tuple[bool, Optional[str]]:
        """Validate integer value is within range."""
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return False, f"{name} must be an integer"
        
        if int_value < min_val:
            return False, f"{name} must be >= {min_val}"
        
        if max_val is not None and int_value > max_val:
            return False, f"{name} must be <= {max_val}"
        
        return True, None

    @staticmethod
    def validate_list_of_strings(value: Any, max_items: Optional[int] = None,
                                 name: str = "list") -> tuple[bool, Optional[str]]:
        """Validate list contains only strings."""
        if not isinstance(value, list):
            return False, f"{name} must be a list"
        
        if not all(isinstance(item, str) for item in value):
            return False, f"All items in {name} must be strings"
        
        if max_items and len(value) > max_items:
            return False, f"{name} cannot have more than {max_items} items"
        
        return True, None

    @staticmethod
    def validate_non_empty_string(value: Any, max_length: Optional[int] = None,
                                 name: str = "string") -> tuple[bool, Optional[str]]:
        """Validate non-empty string."""
        if not isinstance(value, str):
            return False, f"{name} must be a string"
        
        if not value or not value.strip():
            return False, f"{name} cannot be empty"
        
        if max_length and len(value) > max_length:
            return False, f"{name} exceeds maximum length of {max_length}"
        
        return True, None

    @staticmethod
    def sanitize_code(code: str) -> str:
        """Basic code sanitization."""
        return code.strip()

    @staticmethod
    def normalize_language(language: str) -> str:
        """Normalize language name."""
        return language.lower().strip()


class ResultFormatter:
    """Common result formatting to eliminate duplication."""

    @staticmethod
    def format_search_result(index: int, content: str, score: float, metadata: Optional[dict] = None) -> dict:
        """Format a search result in standard way."""
        return {
            "index": index,
            "content": content,
            "score": round(score, 4) if isinstance(score, float) else score,
            "metadata": metadata or {}
        }

    @staticmethod
    def format_batch_results(results: List[Any], status: str = "success") -> dict:
        """Format batch operation results."""
        return {
            "status": status,
            "count": len(results),
            "results": results
        }

    @staticmethod
    def rank_and_format_results(results: List[dict], key: str = "score",
                               descending: bool = True, top_k: Optional[int] = None) -> List[dict]:
        """Rank results by score and optionally limit."""
        sorted_results = sorted(results, key=lambda x: x.get(key, 0), reverse=descending)
        if top_k:
            sorted_results = sorted_results[:top_k]
        return sorted_results
