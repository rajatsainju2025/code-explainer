"""Optimized dependency injection with caching and reduced lock contention."""

import os
import gc
import hashlib
import secrets
from typing import Optional, Dict, Tuple
from fastapi import Depends, HTTPException, Request, Header
from ..model.core import CodeExplainer
from ..config import Config


# Global model instance for dependency injection
_global_explainer: Optional[CodeExplainer] = None
_explainer_lock = __import__('threading').RLock()

# Cache for config instance to avoid repeated creation
_cached_config: Optional[Config] = None
_config_lock = __import__('threading').RLock()

# Cache for API key validation to avoid repeated environment variable access
_api_keys_cache: Optional[Tuple[set, bool]] = None  # (keys_set, auth_required)
_api_keys_lock = __import__('threading').RLock()
_API_KEYS_CACHE_TTL = 300  # 5 minutes
_api_keys_cache_time = 0


def _refresh_api_keys_cache() -> Tuple[set, bool]:
    """Refresh cached API keys from environment."""
    global _api_keys_cache, _api_keys_cache_time
    import time
    
    current_time = time.time()
    
    # Check if cache is still valid
    if _api_keys_cache is not None and (current_time - _api_keys_cache_time) < _API_KEYS_CACHE_TTL:
        return _api_keys_cache
    
    with _api_keys_lock:
        # Double-check locking
        current_time = time.time()
        if _api_keys_cache is not None and (current_time - _api_keys_cache_time) < _API_KEYS_CACHE_TTL:
            return _api_keys_cache
        
        # Build keys set
        configured_keys = os.environ.get('CODE_EXPLAINER_API_KEYS', '')
        single_key = os.environ.get('CODE_EXPLAINER_API_KEY', '')
        
        valid_keys = set()
        if configured_keys:
            valid_keys.update(k.strip() for k in configured_keys.split(',') if k.strip())
        if single_key:
            valid_keys.add(single_key.strip())
        
        # Pre-hash all keys for faster comparison
        hashed_keys = set()
        for key in valid_keys:
            hashed_keys.add(hashlib.sha256(key.encode()).hexdigest())
        
        auth_required = bool(hashed_keys)
        _api_keys_cache = (hashed_keys, auth_required)
        _api_keys_cache_time = current_time
        
        return _api_keys_cache


def get_config() -> Config:
    """Get configuration instance with caching."""
    global _cached_config
    
    # Fast path: if cached config exists, return it
    if _cached_config is not None:
        return _cached_config
    
    with _config_lock:
        if _cached_config is None:
            _cached_config = Config()
        return _cached_config


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
            
        # Ensure model is moved to optimal device only once
        # and reuse tokenizer/model across requests (singleton)
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
                config = get_config()
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
    """Validate API key against configured keys with efficient caching.
    
    Supports multiple API keys stored in:
    1. Environment variable: CODE_EXPLAINER_API_KEYS (comma-separated)
    2. Environment variable: CODE_EXPLAINER_API_KEY (single key)
    3. No keys configured = open access
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if key is valid or no keys configured, False otherwise
    """
    # Get cached keys and auth requirement
    hashed_keys, auth_required = _refresh_api_keys_cache()
    
    # If no keys configured, allow open access
    if not auth_required:
        return True
    
    # If keys are configured but none provided, deny
    if api_key is None:
        return False
    
    # Use constant-time comparison to prevent timing attacks
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return secrets.compare_digest(api_key_hash in hashed_keys, True)


def get_optional_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Get optional API key from headers for authentication.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        API key if provided, None otherwise
    """
    return x_api_key


def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """Require API key for authentication.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        Valid API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not validate_api_key(x_api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return x_api_key
