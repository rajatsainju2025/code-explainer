"""
Custom exception types for the Code Explainer system.

This module defines a hierarchy of exceptions to provide better error handling
and more specific error information throughout the codebase.
"""

from typing import Optional, Dict, Any


class CodeExplainerError(Exception):
    """Base exception for all Code Explainer errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        error_str = self.message
        if self.error_code:
            error_str = f"[{self.error_code}] {error_str}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            error_str = f"{error_str} (Context: {context_str})"
        return error_str


class ConfigurationError(CodeExplainerError):
    """Raised when there are configuration-related errors."""
    
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
            context=context
        )


class ModelError(CodeExplainerError):
    """Raised when there are model loading or inference errors."""
    
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
            context=context
        )


class ValidationError(CodeExplainerError):
    """Raised when input validation fails."""
    
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
            context=context
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