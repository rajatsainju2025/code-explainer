"""FastAPI server for Code Explainer service.

Performance-focused tweaks applied:
- Use ORJSONResponse when available for faster JSON serialization
- Configure gzip compression via middleware (set up in middleware.py)
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    # Prefer orjson if available for high-performance JSON serialization
    from fastapi.responses import ORJSONResponse  # type: ignore
    DefaultJSONResponse = ORJSONResponse
except Exception:  # pragma: no cover - optional dependency
    DefaultJSONResponse = JSONResponse

from .. import __version__
from .middleware import setup_all_middleware
from .endpoints import router as api_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Code Explainer API",
        description="AI-powered code explanation service",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        default_response_class=DefaultJSONResponse,
    )

    # Setup middleware
    setup_all_middleware(app)

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "Code Explainer API",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/v1/health"
        }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    return app


# Create default app instance for testing
app = create_app()


def main():
    """Main entry point for running the server."""
    import argparse
    # Use uvloop if available for faster event loop on *nix platforms
    try:  # pragma: no cover
        import uvloop  # type: ignore
        uvloop.install()
        logger.info("uvloop installed for faster event loop")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Code Explainer API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    app = create_app()

    logger.info(f"Starting Code Explainer API server on {args.host}:{args.port}")
    uvicorn.run(
        "code_explainer.api.server:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True
    )


if __name__ == "__main__":
    main()
