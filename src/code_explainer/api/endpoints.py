"""API endpoints for the Code Explainer service."""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import (
    CodeExplanationRequest,
    CodeExplanationResponse,
    HealthResponse,
    PerformanceMetricsResponse,
    ErrorResponse
)
from .dependencies import (
    get_code_explainer,
    get_config,
    get_request_id,
    get_optional_api_key
)
from ..model.core import CodeExplainer
from ..config import Config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/explain", response_model=CodeExplanationResponse)
async def explain_code(
    request: CodeExplanationRequest,
    background_tasks: BackgroundTasks,
    explainer: CodeExplainer = Depends(get_code_explainer),
    request_id: str = Depends(get_request_id),
    api_key: Optional[str] = Depends(get_optional_api_key)
) -> CodeExplanationResponse:
    """Explain code using the Code Explainer model."""
    try:
        logger.info(f"[{request_id}] Processing explanation request")

        # Validate input
        if not request.code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")

        # Generate explanation
        explanation = explainer.explain_code(
            code=request.code,
            max_length=request.max_length,
            strategy=request.strategy
        )

        response = CodeExplanationResponse(
            explanation=explanation,
            strategy=request.strategy or "vanilla",
            processing_time=0.0,  # TODO: Add timing
            model_name=getattr(explainer, 'model_name', 'unknown')
        )

        logger.info(f"[{request_id}] Explanation generated successfully")
        return response

    except Exception as e:
        logger.error(f"[{request_id}] Error processing explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    explainer: CodeExplainer = Depends(get_code_explainer),
    request_id: str = Depends(get_request_id)
) -> HealthResponse:
    """Health check endpoint."""
    try:
        # Basic health checks
        model_loaded = hasattr(explainer, 'model') and explainer.model is not None

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            version="1.0.0",  # TODO: Get from config
            model_loaded=model_loaded,
            retrieval_ready=False  # TODO: Check retrieval service
        )
    except Exception as e:
        logger.error(f"[{request_id}] Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            model_loaded=False,
            retrieval_ready=False
        )


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_metrics(
    explainer: CodeExplainer = Depends(get_code_explainer),
    request_id: str = Depends(get_request_id),
    api_key: Optional[str] = Depends(get_optional_api_key)
) -> PerformanceMetricsResponse:
    """Get service metrics."""
    try:
        # Get basic metrics
        return PerformanceMetricsResponse(
            total_requests=0,  # TODO: Implement request counting
            average_response_time=0.0,  # TODO: Implement timing
            cache_hit_rate=0.0,  # TODO: Implement cache metrics
            model_inference_time=0.0  # TODO: Implement inference timing
        )
    except Exception as e:
        logger.error(f"[{request_id}] Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@router.get("/config")
async def get_config_info(
    config: Config = Depends(get_config),
    request_id: str = Depends(get_request_id),
    api_key: Optional[str] = Depends(get_optional_api_key)
) -> Dict[str, Any]:
    """Get configuration information."""
    try:
        # Return safe configuration info (no secrets)
        config_info = {
            "model_name": getattr(config, 'model_name', 'unknown'),
            "max_length": getattr(config, 'max_length', 512),
            "temperature": getattr(config, 'temperature', 0.7),
            "device": getattr(config, 'device', 'cpu'),
            "cache_enabled": getattr(config, 'cache_enabled', True),
            "supported_languages": getattr(config, 'supported_languages', ['python'])
        }

        return {
            "config": config_info,
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"[{request_id}] Failed to get config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {str(e)}")


@router.post("/reload")
async def reload_model(
    background_tasks: BackgroundTasks,
    config: Config = Depends(get_config),
    request_id: str = Depends(get_request_id),
    api_key: Optional[str] = Depends(get_optional_api_key)
) -> Dict[str, Any]:
    """Reload the model (admin endpoint)."""
    try:
        # TODO: Implement model reloading
        # This would typically involve:
        # 1. Unloading current model
        # 2. Reloading with new config
        # 3. Updating the dependency injection

        logger.info(f"[{request_id}] Model reload requested")
        return {
            "status": "reload_initiated",
            "request_id": request_id,
            "message": "Model reload functionality not yet implemented"
        }
    except Exception as e:
        logger.error(f"[{request_id}] Model reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")