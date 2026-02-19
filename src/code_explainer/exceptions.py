"""
Custom exception types for the Code Explainer system.

This module defines a hierarchy of exceptions with:
- __slots__ for memory efficiency
- HTTP status code mapping for API use
- Pickleable for multiprocessing support
- Lazy string formatting
"""

from typing import Optional, Dict, Any, ClassVar


class CodeExplainerError(Exception):
    """Base exception for all Code Explainer errors."""

    __slots__ = ('message', 'error_code', '_context')
    http_status: ClassVar[int] = 500
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self._context = context
    
    @property
    def context(self) -> Dict[str, Any]:
        """Get context dict, lazily initializing if None."""
        if self._context is None:
            self._context = {}
        return self._context
    
    def __str__(self) -> str:
        """Lazy string formatting - only format when actually printed."""
        if not self.error_code and not self._context:
            # Fast path: just return message
            return self.message
            
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
        
        return " ".join(parts)
    
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