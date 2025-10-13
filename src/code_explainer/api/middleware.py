"""Middleware setup for FastAPI application."""

import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to each request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        logger.info(f"Request: {request.method} {request.url}")

        response = await call_next(request)

        logger.info(f"Response: {response.status_code}")
        return response


def setup_cors_middleware(app, allowed_origins=None):
    """Setup CORS middleware."""
    if allowed_origins is None:
        allowed_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_rate_limiting(app):
    """Setup rate limiting if available."""
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.util import get_remote_address

        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(429, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)

        logger.info("Rate limiting enabled")
    except ImportError:
        logger.warning("SlowAPI not available, rate limiting disabled")


def setup_all_middleware(app):
    """Setup all middleware for the application."""
    # Add request ID middleware
    app.add_middleware(RequestIDMiddleware)

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)

    # Setup CORS
    setup_cors_middleware(app)

    # Setup rate limiting
    setup_rate_limiting(app)