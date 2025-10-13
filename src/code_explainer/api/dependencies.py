"""Dependency injection for FastAPI application."""

from typing import Optional
from fastapi import Depends, HTTPException, Request
from ..model.core import CodeExplainer
from ..config import Config


def get_config() -> Config:
    """Get configuration instance."""
    return Config()


def get_code_explainer(config: Config = Depends(get_config)) -> CodeExplainer:
    """Get CodeExplainer instance with dependency injection."""
    try:
        return CodeExplainer(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, 'request_id', 'unknown')


def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Validate API key if provided."""
    if api_key is None:
        return True  # Allow requests without API key for now

    # TODO: Implement proper API key validation
    # For now, accept any non-empty key
    return bool(api_key.strip())


def get_optional_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Get optional API key for authentication."""
    if api_key and validate_api_key(api_key):
        return api_key
    return None