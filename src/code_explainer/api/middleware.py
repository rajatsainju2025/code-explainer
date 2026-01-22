"""Middleware setup for FastAPI application."""

import time
import uuid
import logging
from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

logger = logging.getLogger(__name__)

# Pre-create UUID factory for faster request ID generation
_UUID4 = uuid.uuid4

# Pre-compile common response headers
_HEADER_REQUEST_ID = "X-Request-ID"
_HEADER_RESPONSE_TIME = "X-Response-Time"
_HEADER_SERVER_TIMING = "Server-Timing"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add unique request ID to each request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Try to get request ID from headers, otherwise generate new one
        request_id = request.headers.get(_HEADER_REQUEST_ID) or str(_UUID4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[_HEADER_REQUEST_ID] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for structured request/response logging with timing."""
    
    # Skip logging for these paths to reduce noise (frozenset for O(1) lookup)
    _SKIP_PATHS = frozenset({'/health', '/metrics', '/favicon.ico', '/docs', '/redoc', '/openapi.json'})
    
    # Cache perf_counter function reference for faster access
    __slots__ = ()
    _perf_counter = time.perf_counter

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for health checks and metrics (high frequency, low value)
        path = request.url.path
        if path in self._SKIP_PATHS:
            return await call_next(request)
        
        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = self._perf_counter()  # More precise than time.time()
        
        # Log incoming request (only if not skipped)
        if logger.isEnabledFor(logging.INFO):
            client_host = request.client.host if request.client else 'unknown'
            logger.info(\"[%s] %s %s from %s\", request_id, request.method, path, client_host)

        try:
            response = await call_next(request)
            duration = self._perf_counter() - start_time
            
            # Log response with timing
            if logger.isEnabledFor(logging.INFO):\n                logger.info(\"[%s] %s completed in %.4fs\", request_id, response.status_code, duration)
            
            # Add timing headers
            response.headers[_HEADER_RESPONSE_TIME] = f\"{duration:.4f}\"
            response.headers[_HEADER_SERVER_TIMING] = f\"app;dur={duration*1000:.1f}\"
            
            return response
        except Exception as e:
            duration = self._perf_counter() - start_time
            logger.error(\"[%s] Request failed after %.4fs: %s\", request_id, duration, str(e), exc_info=True)
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware to catch and format exceptions."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        try:
            return await call_next(request)
        except ValueError as e:
            logger.warning("[%s] Validation error: %s", request_id, str(e))
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Validation Error",
                    "detail": str(e),
                    "request_id": request_id
                }
            )
        except PermissionError as e:
            logger.warning("[%s] Permission denied: %s", request_id, str(e))
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Permission Denied",
                    "detail": str(e),
                    "request_id": request_id
                }
            )
        except FileNotFoundError as e:
            logger.warning("[%s] Resource not found: %s", request_id, str(e))
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "Not Found",
                    "detail": str(e),
                    "request_id": request_id
                }
            )
        except Exception as e:
            logger.error(
                "[%s] Unhandled exception: %s", request_id, str(e),
                exc_info=True
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "detail": "An unexpected error occurred. Check logs for details.",
                    "request_id": request_id
                }
            )


def setup_cors_middleware(app, allowed_origins=None):
    """Setup CORS middleware with security best practices.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins, defaults to environment variable
    """
    import os
    
    if allowed_origins is None:
        # Get from environment or use safe default
        origins_env = os.environ.get('CORS_ALLOWED_ORIGINS', '')
        if origins_env:
            allowed_origins = [o.strip() for o in origins_env.split(',')]
        else:
            # Development default - allow all
            # In production, set CORS_ALLOWED_ORIGINS env variable
            allowed_origins = ["*"]
            logger.warning(
                "CORS allowing all origins (*). "
                "Set CORS_ALLOWED_ORIGINS env variable in production."
            )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
        max_age=3600,  # Cache preflight requests for 1 hour
    )
    
    logger.info("CORS configured with origins: %s", allowed_origins)


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
    """Setup all middleware for the application in correct order.
    
    Middleware is applied in reverse order, so we add:
    1. Error handling (outermost)
    2. Request ID
    3. Logging
    4. CORS
    5. Rate limiting (if available)
    """
    # Setup rate limiting first (innermost)
    setup_rate_limiting(app)
    
    # Setup CORS
    setup_cors_middleware(app)

    # Add gzip compression for large responses (saves bandwidth, faster over network)
    try:
        app.add_middleware(GZipMiddleware, minimum_size=1024)
        logger.info("GZip compression enabled (min_size=1024 bytes)")
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to enable GZipMiddleware: %s", e)

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Add request ID middleware
    app.add_middleware(RequestIDMiddleware)
    
    # Add error handling middleware (outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    
    logger.info("All middleware configured successfully")