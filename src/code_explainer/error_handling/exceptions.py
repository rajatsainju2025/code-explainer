"""Custom exception hierarchy for Code Explainer."""

# Custom exception hierarchy
class CodeExplainerError(Exception):
    """Base exception for Code Explainer."""
    pass

class ModelError(CodeExplainerError):
    """Exception raised for model-related errors."""
    pass

class ConfigurationError(CodeExplainerError):
    """Exception raised for configuration-related errors."""
    pass

class ValidationError(CodeExplainerError):
    """Exception raised for input validation errors."""
    pass

class ProcessingError(CodeExplainerError):
    """Exception raised for processing-related errors."""
    pass

class ResourceError(CodeExplainerError):
    """Exception raised for resource-related errors."""
    pass