"""Pydantic models for API endpoints."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class CodeExplanationRequest(BaseModel):
    """Request model for code explanation."""
    model_config = ConfigDict(extra="ignore", frozen=False)
    code: str = Field(..., description="Python code to explain", min_length=1)
    strategy: Optional[str] = Field(
        default="vanilla",
        description="Explanation strategy"
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum length of generated explanation",
        ge=1,
        le=4096
    )
    include_symbolic: Optional[bool] = Field(
        default=False,
        description="Include symbolic analysis in explanation"
    )
    include_retrieval: Optional[bool] = Field(
        default=False,
        description="Include retrieval-augmented generation"
    )


class BatchCodeExplanationRequest(BaseModel):
    """Request model for batch code explanation."""
    requests: List[Dict[str, Any]] = Field(
        ...,
        description="List of explanation requests"
    )


class CodeExplanationResponse(BaseModel):
    """Response model for code explanation."""
    model_config = ConfigDict(extra="ignore", ser_json_inf_nan="null")
    explanation: str = Field(..., description="Generated explanation")
    strategy: str = Field(..., description="Strategy used")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_name: Optional[str] = Field(None, description="Model used")


class HealthResponse(BaseModel):
    """Response model for health check."""
    model_config = ConfigDict(extra="ignore")
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    retrieval_ready: Optional[bool] = Field(None, description="Whether retrieval service is ready")


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    model_config = ConfigDict(extra="ignore")
    total_requests: int = Field(..., description="Total requests processed")
    average_response_time: float = Field(..., description="Average response time in seconds")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    model_inference_time: float = Field(..., description="Average model inference time")


class ErrorResponse(BaseModel):
    """Error response model."""
    model_config = ConfigDict(extra="ignore")
    detail: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Error code")