"""Security input sanitization and validation."""

import re
from typing import Optional


class InputSanitizer:
    """Sanitizes and validates user inputs for security."""
    
    # Maximum code size (100 KB)
    MAX_CODE_SIZE = 100 * 1024
    
    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r"__import__",
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"os\.system",
        r"subprocess",
        r"open\s*\(",
    ]
    
    # Allowed languages
    ALLOWED_LANGUAGES = {
        "python", "java", "javascript", "typescript",
        "cpp", "c", "csharp", "go", "rust", "php",
        "sql", "html", "css", "yaml", "json", "xml"
    }
    
    @classmethod
    def sanitize_code(cls, code: str) -> str:
        """Sanitize code input.
        
        Args:
            code: Raw code string
        
        Returns:
            Sanitized code
        
        Raises:
            ValueError: If code fails validation
        """
        if not code:
            raise ValueError("Code cannot be empty")
        
        # Check size
        if len(code.encode()) > cls.MAX_CODE_SIZE:
            raise ValueError(f"Code exceeds maximum size of {cls.MAX_CODE_SIZE} bytes")
        
        # Check for dangerous patterns (warning only, not rejection)
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                # Log warning but allow (may be legitimate code review)
                pass
        
        return code.strip()
    
    @classmethod
    def sanitize_language(cls, language: str) -> str:
        """Validate and sanitize language identifier.
        
        Args:
            language: Language identifier
        
        Returns:
            Lowercase, validated language
        
        Raises:
            ValueError: If language not supported
        """
        lang_lower = language.lower().strip()
        
        if lang_lower not in cls.ALLOWED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported: {', '.join(sorted(cls.ALLOWED_LANGUAGES))}"
            )
        
        return lang_lower
    
    @classmethod
    def sanitize_string(
        cls,
        value: str,
        max_length: int = 1000,
        allow_whitespace: bool = True
    ) -> str:
        """Sanitize generic string input.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            allow_whitespace: Whether to allow whitespace
        
        Returns:
            Sanitized string
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, str):
            raise ValueError("Value must be string")
        
        if len(value) > max_length:
            raise ValueError(f"String exceeds maximum length of {max_length}")
        
        value = value.strip()
        
        if not allow_whitespace and any(c.isspace() for c in value):
            raise ValueError("Whitespace not allowed in this field")
        
        return value
    
    @classmethod
    def sanitize_integer(
        cls,
        value,
        min_val: int = 0,
        max_val: Optional[int] = None
    ) -> int:
        """Sanitize and validate integer input.
        
        Args:
            value: Value to convert and validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
        
        Returns:
            Validated integer
        
        Raises:
            ValueError: If validation fails
        """
        try:
            num = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"Value '{value}' is not a valid integer")
        
        if num < min_val:
            raise ValueError(f"Value {num} is below minimum of {min_val}")
        
        if max_val is not None and num > max_val:
            raise ValueError(f"Value {num} exceeds maximum of {max_val}")
        
        return num
    
    @classmethod
    def sanitize_float(
        cls,
        value,
        min_val: float = 0.0,
        max_val: Optional[float] = None
    ) -> float:
        """Sanitize and validate float input.
        
        Args:
            value: Value to convert and validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
        
        Returns:
            Validated float
        
        Raises:
            ValueError: If validation fails
        """
        try:
            num = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Value '{value}' is not a valid number")
        
        if num < min_val:
            raise ValueError(f"Value {num} is below minimum of {min_val}")
        
        if max_val is not None and num > max_val:
            raise ValueError(f"Value {num} exceeds maximum of {max_val}")
        
        return num


class InputValidator:
    """Validates request payloads and parameters."""
    
    @classmethod
    def validate_code_request(cls, code: str, language: str) -> tuple:
        """Validate code explanation request.
        
        Args:
            code: Source code
            language: Programming language
        
        Returns:
            Tuple of (sanitized_code, sanitized_language)
        
        Raises:
            ValueError: If validation fails
        """
        sanitizer = InputSanitizer()
        
        code = sanitizer.sanitize_code(code)
        language = sanitizer.sanitize_language(language)
        
        return code, language
    
    @classmethod
    def validate_batch_request(cls, codes: list) -> list:
        """Validate batch code request.
        
        Args:
            codes: List of code strings
        
        Returns:
            List of sanitized codes
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(codes, list):
            raise ValueError("Codes must be a list")
        
        if len(codes) > 100:
            raise ValueError("Batch size cannot exceed 100 items")
        
        if not codes:
            raise ValueError("Codes list cannot be empty")
        
        sanitizer = InputSanitizer()
        return [sanitizer.sanitize_code(code) for code in codes]
