"""Dependency injection for FastAPI application."""

import os
import gc
import hashlib
import secrets
from typing import Optional
from fastapi import Depends, HTTPException, Request, Header
from ..model.core import CodeExplainer
from ..config import Config


# Global model instance for dependency injection
_global_explainer: Optional[CodeExplainer] = None
_explainer_lock = __import__('threading').RLock()


def get_config() -> Config:
    """Get configuration instance."""
    return Config()


def get_code_explainer(config: Config = Depends(get_config)) -> CodeExplainer:
    """Get CodeExplainer instance with dependency injection.
    
    Uses a global singleton pattern to avoid reloading the model
    for every request.
    """
    global _global_explainer
    
    with _explainer_lock:
        if _global_explainer is None:
            try:
                _global_explainer = CodeExplainer(config)
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to initialize model: {str(e)}"
                )
        
        return _global_explainer


def reload_code_explainer(config: Optional[Config] = None) -> CodeExplainer:
    """Reload the CodeExplainer model.
    
    Args:
        config: Optional new configuration to use
        
    Returns:
        New CodeExplainer instance
        
    Raises:
        HTTPException: If model reload fails
    """
    global _global_explainer
    
    with _explainer_lock:
        # Cleanup old model
        if _global_explainer is not None:
            try:
                # Try to cleanup memory
                if hasattr(_global_explainer, 'cleanup_memory'):
                    _global_explainer.cleanup_memory()
                del _global_explainer
                gc.collect()
            except Exception as e:
                # Log but don't fail if cleanup has issues
                import logging
                logging.getLogger(__name__).warning(
                    f"Model cleanup had issues: {str(e)}"
                )
        
        # Load new model
        try:
            if config is None:
                config = Config()
            _global_explainer = CodeExplainer(config)
            return _global_explainer
        except Exception as e:
            _global_explainer = None
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reload model: {str(e)}"
            )


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, 'request_id', 'unknown')


def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Validate API key against configured keys.
    
    Supports multiple API keys stored in:
    1. Environment variable: CODE_EXPLAINER_API_KEYS (comma-separated)
    2. Environment variable: CODE_EXPLAINER_API_KEY (single key)
    3. No keys configured = open access
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if key is valid or no keys configured, False otherwise
    """
    if api_key is None:
        # Check if API keys are configured
        configured_keys = os.environ.get('CODE_EXPLAINER_API_KEYS', '')
        single_key = os.environ.get('CODE_EXPLAINER_API_KEY', '')
        
        # If no keys configured, allow open access
        if not configured_keys and not single_key:
            return True
        
        # If keys are configured, require authentication
        return False
    
    # Get configured API keys
    configured_keys = os.environ.get('CODE_EXPLAINER_API_KEYS', '')
    single_key = os.environ.get('CODE_EXPLAINER_API_KEY', '')
    
    # Build list of valid keys
    valid_keys = set()
    if configured_keys:
        valid_keys.update(k.strip() for k in configured_keys.split(',') if k.strip())
    if single_key:
        valid_keys.add(single_key.strip())
    
    # If no keys configured, allow access
    if not valid_keys:
        return True
    
    # Validate using constant-time comparison to prevent timing attacks
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    for valid_key in valid_keys:
        valid_key_hash = hashlib.sha256(valid_key.encode()).hexdigest()
        if secrets.compare_digest(api_key_hash, valid_key_hash):
            return True
    
    return False


def get_optional_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Get optional API key from headers for authentication.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        API key if valid, None otherwise
    """
    if x_api_key and validate_api_key(x_api_key):
        return x_api_key
    return None


def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """Require valid API key for protected endpoints.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not validate_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    return x_api_key or ""