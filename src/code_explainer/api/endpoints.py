"""API endpoints for the Code Explainer service."""

import time
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
    get_optional_api_key,
    reload_code_explainer,
    require_api_key
)
from .metrics import get_metrics_collector
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
    metrics_collector = get_metrics_collector()
    request_metrics = metrics_collector.start_request(request_id, "/explain")
    
    try:
        logger.info(f"[{request_id}] Processing explanation request")

        # Validate input
        if not request.code.strip():
            metrics_collector.end_request(request_metrics, status_code=400)
            raise HTTPException(status_code=400, detail="Code cannot be empty")

        # Track if cache is used
        start_inference = time.time()
        
        # Check if result is cached
        cache_key = None
        if hasattr(explainer, 'explanation_cache') and explainer.explanation_cache:
            from ..cache.utils import generate_cache_key
            cache_key = generate_cache_key(
                request.code,
                request.strategy or "vanilla",
                getattr(explainer, 'model_name', 'unknown')
            )
            if explainer.explanation_cache.get(
                request.code,
                request.strategy or "vanilla", 
                getattr(explainer, 'model_name', 'unknown')
            ):
                metrics_collector.record_cache_hit()
            else:
                metrics_collector.record_cache_miss()
        
        # Generate explanation
        explanation = explainer.explain_code(
            code=request.code,
            max_length=request.max_length,
            strategy=request.strategy
        )
        
        # Record inference time
        inference_time = time.time() - start_inference
        metrics_collector.record_model_inference(inference_time)

        processing_time = time.time() - request_metrics.start_time
        
        response = CodeExplanationResponse(
            explanation=explanation,
            strategy=request.strategy or "vanilla",
            processing_time=round(processing_time, 4),
            model_name=getattr(explainer, 'model_name', 'unknown')
        )

        metrics_collector.end_request(request_metrics, status_code=200)
        logger.info(f"[{request_id}] Explanation generated in {processing_time:.4f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.end_request(request_metrics, status_code=500, error=str(e))
        logger.error(f"[{request_id}] Error processing explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    explainer: CodeExplainer = Depends(get_code_explainer),
    request_id: str = Depends(get_request_id)
) -> HealthResponse:
    """Health check endpoint."""
    try:
        # Import here to get version
        from .. import __version__
        
        # Basic health checks
        model_loaded = hasattr(explainer, 'model') and explainer.model is not None
        
        # Check retrieval service readiness
        retrieval_ready = False
        if hasattr(explainer, 'retrieval_service') and explainer.retrieval_service:
            try:
                retrieval_ready = explainer.retrieval_service.is_ready()
            except:
                retrieval_ready = False

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            version=__version__,
            model_loaded=model_loaded,
            retrieval_ready=retrieval_ready
        )
    except Exception as e:
        logger.error(f"[{request_id}] Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="unknown",
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
        metrics_collector = get_metrics_collector()
        metrics = metrics_collector.get_metrics()
        
        return PerformanceMetricsResponse(
            total_requests=metrics["total_requests"],
            average_response_time=metrics["average_response_time"],
            cache_hit_rate=metrics["cache_hit_rate"],
            model_inference_time=metrics["model_inference_time"]
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


@router.get("/version")
async def get_version(
    config: Config = Depends(get_config),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get detailed version information."""
    try:
        from .. import __version__, __author__
        import sys
        import torch
        from transformers import __version__ as transformers_version
        
        version_info = {
            "code_explainer_version": __version__,
            "author": __author__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "device": getattr(config, 'device', 'cpu'),
            "request_id": request_id
        }
        
        # Add CUDA info if available
        if torch.cuda.is_available():
            version_info["cuda_version"] = torch.version.cuda
            version_info["cudnn_version"] = torch.backends.cudnn.version()
        
        return version_info
    except Exception as e:
        logger.error(f"[{request_id}] Failed to get version info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Version info retrieval failed: {str(e)}")


@router.post("/admin/reload")
async def reload_model(
    background_tasks: BackgroundTasks,
    config: Config = Depends(get_config),
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """Reload the model (admin endpoint - requires API key).
    
    This endpoint allows hot-reloading of the model without restarting the service.
    Useful for applying configuration changes or updating to a new model version.
    """
    try:
        logger.info(f"[{request_id}] Model reload requested by authenticated user")
        
        def reload_in_background():
            """Perform reload in background to avoid blocking."""
            try:
                logger.info("Starting model reload...")
                reload_code_explainer(config)
                logger.info("Model reload completed successfully")
            except Exception as e:
                logger.error(f"Background model reload failed: {str(e)}")
        
        # Queue reload in background
        background_tasks.add_task(reload_in_background)
        
        return {
            "status": "reload_initiated",
            "request_id": request_id,
            "message": "Model reload has been initiated in the background. Check logs for completion status."
        }
    except Exception as e:
        logger.error(f"[{request_id}] Model reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")