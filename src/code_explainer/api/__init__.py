"""API module for Code Explainer.

This module contains the FastAPI server implementation and API endpoints
for code explanation services.
"""

from .server import app

__all__ = ["app"]