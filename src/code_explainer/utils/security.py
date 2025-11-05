"""Security utilities for input validation and sanitization.

Provides functions for securing user inputs and preventing common attacks.
"""

import re
from pathlib import Path
from typing import Optional, Any
from urllib.parse import quote, unquote


def sanitize_code_input(code: str, max_length: int = 100000) -> str:
    """Sanitize code input for safety.
    
    Args:
        code: Code to sanitize
        max_length: Maximum allowed code length
        
    Returns:
        Sanitized code
        
    Raises:
        ValueError: If code exceeds max length
    """
    if len(code) > max_length:
        raise ValueError(f"Code exceeds maximum length of {max_length}")
    return code.strip()


def validate_file_path(path: str, allowed_dirs: Optional[list] = None) -> Path:
    """Validate file path to prevent directory traversal.
    
    Args:
        path: Path to validate
        allowed_dirs: Optional list of allowed directories
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or outside allowed directories
    """
    try:
        p = Path(path).resolve()
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid file path: {e}")
    
    # Check for traversal attacks
    if ".." in str(p):
        raise ValueError("Path traversal detected")
    
    if allowed_dirs:
        if not any(p.is_relative_to(Path(d).resolve()) for d in allowed_dirs):
            raise ValueError("Path is outside allowed directories")
    
    return p


def escape_code_for_display(code: str) -> str:
    """Escape code for safe display.
    
    Args:
        code: Code to escape
        
    Returns:
        Escaped code safe for display
    """
    # Escape HTML-like characters
    escape_map = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '&': '&amp;'
    }
    result = code
    for char, escape in escape_map.items():
        result = result.replace(char, escape)
    return result


def validate_identifier(identifier: str) -> bool:
    """Validate Python identifier.
    
    Args:
        identifier: String to validate as identifier
        
    Returns:
        True if valid Python identifier
    """
    if not identifier:
        return False
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier))


def rate_limit_check(key: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
    """Check if request should be rate limited.
    
    Note: This is a simple in-memory implementation. For production,
    use Redis or similar distributed cache.
    
    Args:
        key: Rate limit key (e.g., IP address, user ID)
        max_requests: Max requests allowed
        window_seconds: Time window in seconds
        
    Returns:
        True if request should be allowed, False if rate limited
    """
    # Placeholder - implementation would use distributed cache
    return True


# Security constants
DANGEROUS_IMPORTS = frozenset({
    'os', 'subprocess', 'sys', 'socket', '__import__',
    'exec', 'eval', 'compile', 'open', '__builtins__'
})

DANGEROUS_FUNCTIONS = frozenset({
    'globals', 'locals', 'vars', 'dir', 'getattr',
    'setattr', 'delattr', '__import__'
})

DANGEROUS_PATTERNS = re.compile(
    r'(import\s+os|import\s+subprocess|exec\(|eval\()',
    re.IGNORECASE
)
