"""
Custom exception types for the Code Explainer system.

This module defines a hierarchy of exceptions to provide better error handling
and more specific error information throughout the codebase.

Optimized for:
- Lazy string formatting to avoid allocation when exception not printed
- __slots__ for memory efficiency
- HTTP status code mapping for API use
- Pickleable for multiprocessing support
"""

from typing import Optional, Dict, Any, ClassVar


class CodeExplainerError(Exception):
    """Base exception for all Code Explainer errors.
    
    Uses lazy string formatting and __slots__ for efficiency.
    """
    
    __slots__ = ('message', 'error_code', '_context', '_str_cached')
    
    # Default HTTP status code for API responses
    http_status: ClassVar[int] = 500
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        # Don't call super().__init__ with formatted string - defer to __str__
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self._context = context  # Use private to allow lazy initialization
        self._str_cached: Optional[str] = None  # Cache formatted string
    
    @property
    def context(self) -> Dict[str, Any]:
        """Get context dict, lazily initializing if None."""
        if self._context is None:
            self._context = {}
        return self._context
    
    def __str__(self) -> str:
        """Lazy string formatting - only format when actually printed."""
        if self._str_cached is not None:
            return self._str_cached
        
        parts = []
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        parts.append(self.message)
        
        if self._context:
            # Limit context items and value length for safety
            context_items = list(self._context.items())[:5]
            context_str = ", ".join(
                f"{k}={str(v)[:50]}" for k, v in context_items
            )
            parts.append(f"(Context: {context_str})")
        
        self._str_cached = " ".join(parts)
        return self._str_cached
    
    def __reduce__(self):
        """Support pickling for multiprocessing."""
        return (
            self.__class__,
            (self.message, self.error_code, self._context)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self._context or {},
            "http_status": self.http_status
        }


class ConfigurationError(CodeExplainerError):
    """Raised when there are configuration-related errors."""
    
    __slots__ = ()
    http_status: ClassVar[int] = 400
    
    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        missing_key: Optional[str] = None
    ) -> None:
        context = {}
        if config_path:
            context["config_path"] = config_path
        if missing_key:
            context["missing_key"] = missing_key
        
        super().__init__(
            message, 
            error_code="CONFIG_ERROR",
            context=context if context else None
        )


class ModelError(CodeExplainerError):
    """Raised when there are model loading or inference errors."""
    
    __slots__ = ()
    http_status: ClassVar[int] = 503
    
    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> None:
        context = {}
        if model_path:
            context["model_path"] = model_path
        if model_type:
            context["model_type"] = model_type
            
        super().__init__(
            message,
            error_code="MODEL_ERROR", 
            context=context if context else None
        )


class ValidationError(CodeExplainerError):
    """Raised when input validation fails."""
    
    __slots__ = ()
    http_status: ClassVar[int] = 422
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[str] = None
    ) -> None:
        context = {}
        if field_name:
            context["field"] = field_name
        if field_value:
            context["value"] = str(field_value)[:100]  # Truncate long values
            
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            context=context if context else None
        )


class CacheError(CodeExplainerError):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        message: str,
        cache_type: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        context = {}
        if cache_type:
            context["cache_type"] = cache_type
        if operation:
            context["operation"] = operation
            
        super().__init__(
            message,
            error_code="CACHE_ERROR",
            context=context
        )


class ResourceError(CodeExplainerError):
    """Raised when resource allocation or management fails."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ) -> None:
        context = {}
        if resource_type:
            context["resource_type"] = resource_type
        if resource_id:
            context["resource_id"] = resource_id
            
        super().__init__(
            message,
            error_code="RESOURCE_ERROR",
            context=context
        )


class InferenceError(CodeExplainerError):
    """Raised when code explanation inference fails."""
    
    def __init__(
        self,
        message: str,
        strategy: Optional[str] = None,
        code_length: Optional[int] = None
    ) -> None:
        context = {}
        if strategy:
            context["strategy"] = strategy
        if code_length:
            context["code_length"] = code_length
            
        super().__init__(
            message,
            error_code="INFERENCE_ERROR",
            context=context
        )


class TimeoutError(CodeExplainerError):
    """Raised when operations exceed timeout limits."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None
    ) -> None:
        context = {}
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        if operation:
            context["operation"] = operation
            
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            context=context
        )


class SecurityError(CodeExplainerError):
    """Raised when security checks fail."""
    
    def __init__(
        self,
        message: str,
        security_check: Optional[str] = None,
        code_snippet: Optional[str] = None
    ) -> None:
        context = {}
        if security_check:
            context["security_check"] = security_check
        if code_snippet:
            # Only include first 50 chars for security
            context["code_snippet"] = code_snippet[:50] + "..." if len(code_snippet) > 50 else code_snippet
            
        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            context=context
        )


# Backwards compatibility aliases
ConfigError = ConfigurationError
InferenceFailure = InferenceError