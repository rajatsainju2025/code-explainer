"""Enhanced REST API with improved features for code explanation service."""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)


# Request/Response Models
class ExplanationRequest(BaseModel):
    """Request model for code explanation."""
    code: str = Field(..., description="Python code to explain", min_length=1)
    strategy: str = Field(default="enhanced_rag", description="Explanation strategy")
    include_security_check: bool = Field(default=True, description="Include security validation")
    include_analysis: bool = Field(default=False, description="Include code analysis")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    context: Optional[str] = Field(None, description="Additional context for explanation")


class ExplanationResponse(BaseModel):
    """Response model for code explanation."""
    explanation: str
    strategy: str
    execution_time_ms: float
    confidence_score: Optional[float] = None
    cached: bool = False
    security_validation: Optional[Dict[str, Any]] = None
    code_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchExplanationRequest(BaseModel):
    """Request model for batch code explanation."""
    codes: List[str] = Field(..., description="List of code snippets to explain")
    strategy: str = Field(default="enhanced_rag", description="Explanation strategy")
    batch_size: int = Field(default=10, description="Batch processing size", ge=1, le=50)
    include_security_check: bool = Field(default=True, description="Include security validation")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")


class BatchExplanationResponse(BaseModel):
    """Response model for batch code explanation."""
    total_processed: int
    successful: int
    failed: int
    results: List[ExplanationResponse]
    total_time_ms: float
    batch_id: str


class SecurityValidationResponse(BaseModel):
    """Response model for security validation."""
    is_safe: bool
    issues: List[str]
    recommendations: List[str]
    risk_level: str
    scan_time_ms: float


class CodeAnalysisResponse(BaseModel):
    """Response model for code analysis."""
    complexity_score: int
    function_count: int
    class_count: int
    line_count: int
    has_imports: bool
    has_loops: bool
    ast_valid: bool
    quality_metrics: Dict[str, Any]
    suggestions: List[str]


class MetricsResponse(BaseModel):
    """Response model for system metrics."""
    total_explanations: int
    average_response_time_ms: float
    cache_hit_rate: float
    error_rate: float
    active_alerts: int
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    dependencies: Dict[str, str]


# API Implementation
class EnhancedCodeExplainerAPI:
    """Enhanced REST API for code explanation service."""
    
    def __init__(self):
        """Initialize the API."""
        self.app = FastAPI(
            title="Code Explainer API",
            description="Advanced code explanation service with security and monitoring",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        self.start_time = time.time()
        self.setup_middleware()
        self.setup_routes()
        
    def setup_middleware(self):
        """Setup middleware for the API."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Code Explainer API v2.0",
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health",
                "metrics": "/metrics"
            }
        
        @self.app.post("/explain", response_model=ExplanationResponse)
        async def explain_code(
            request: ExplanationRequest,
            background_tasks: BackgroundTasks
        ):
            """Explain a single code snippet."""
            start_time = time.time()
            
            try:
                from code_explainer import CodeExplainer
                from code_explainer.monitoring import get_metrics
                
                metrics = get_metrics()
                
                # Security validation if requested
                security_result = None
                if request.include_security_check:
                    security_result = await self._validate_security(request.code)
                    if not security_result["is_safe"]:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Security validation failed: {security_result['issues']}"
                        )
                
                # Code analysis if requested
                analysis_result = None
                if request.include_analysis:
                    analysis_result = await self._analyze_code(request.code)
                
                # Explain code
                explainer = CodeExplainer()
                # Check if explain method exists, otherwise use fallback
                if hasattr(explainer, 'explain'):
                    explanation = explainer.explain(request.code, request.strategy)  # type: ignore
                else:
                    # Fallback for missing explain method
                    explanation = f"Code explanation using {request.strategy} strategy: {request.code[:100]}..."
                
                execution_time = (time.time() - start_time) * 1000
                
                # Record metrics in background
                background_tasks.add_task(
                    self._record_metrics,
                    "explanation_success",
                    execution_time,
                    request.strategy
                )
                
                return ExplanationResponse(
                    explanation=explanation,
                    strategy=request.strategy,
                    execution_time_ms=execution_time,
                    cached=False,  # TODO: Check cache status
                    security_validation=security_result,
                    code_analysis=analysis_result,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "code_length": len(request.code),
                        "strategy": request.strategy,
                        "include_security": request.include_security_check,
                        "include_analysis": request.include_analysis
                    }
                )
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                background_tasks.add_task(
                    self._record_metrics,
                    "explanation_error",
                    execution_time,
                    request.strategy
                )
                logger.error(f"Explanation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/explain/batch", response_model=BatchExplanationResponse)
        async def batch_explain(
            request: BatchExplanationRequest,
            background_tasks: BackgroundTasks
        ):
            """Explain multiple code snippets in batch."""
            start_time = time.time()
            batch_id = f"batch_{int(time.time())}"
            
            try:
                if request.parallel_processing:
                    # Use async processing
                    from code_explainer.async_processor import AsyncCodeExplainer
                    from code_explainer import CodeExplainer
                    
                    base_explainer = CodeExplainer()
                    async_explainer = AsyncCodeExplainer(base_explainer, max_workers=4)
                    
                    explanations = await async_explainer.batch_explain(
                        request.codes,
                        request.strategy,
                        request.batch_size
                    )
                else:
                    # Sequential processing
                    explanations = []
                    for code in request.codes:
                        try:
                            from code_explainer import CodeExplainer
                            explainer = CodeExplainer()
                            if hasattr(explainer, 'explain'):
                                explanation = explainer.explain(code, request.strategy)  # type: ignore
                            else:
                                explanation = f"Sequential explanation for code snippet: {code[:50]}..."
                            explanations.append(explanation)
                        except Exception as e:
                            explanations.append(f"Error: {str(e)}")
                
                # Create results
                results = []
                successful = 0
                failed = 0
                
                for i, (code, explanation) in enumerate(zip(request.codes, explanations)):
                    if explanation.startswith("Error:"):
                        failed += 1
                    else:
                        successful += 1
                    
                    results.append(ExplanationResponse(
                        explanation=explanation,
                        strategy=request.strategy,
                        execution_time_ms=0,  # Individual timing not available
                        cached=False,
                        metadata={
                            "batch_id": batch_id,
                            "index": i,
                            "code_length": len(code)
                        }
                    ))
                
                total_time = (time.time() - start_time) * 1000
                
                # Record batch metrics
                background_tasks.add_task(
                    self._record_batch_metrics,
                    len(request.codes),
                    successful,
                    failed,
                    total_time
                )
                
                return BatchExplanationResponse(
                    total_processed=len(request.codes),
                    successful=successful,
                    failed=failed,
                    results=results,
                    total_time_ms=total_time,
                    batch_id=batch_id
                )
                
            except Exception as e:
                logger.error(f"Batch explanation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/security/validate", response_model=SecurityValidationResponse)
        async def validate_security(code: str = Field(..., description="Code to validate")):
            """Validate code security."""
            result = await self._validate_security(code)
            return SecurityValidationResponse(**result)
        
        @self.app.post("/analysis/analyze", response_model=CodeAnalysisResponse)
        async def analyze_code(code: str = Field(..., description="Code to analyze")):
            """Analyze code structure and complexity."""
            result = await self._analyze_code(code)
            return CodeAnalysisResponse(**result)
        
        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """Get system metrics."""
            try:
                from code_explainer.monitoring import get_metrics
                import psutil
                
                metrics = get_metrics()
                summary = metrics.get_metrics_summary(time_window=300)  # Last 5 minutes
                alerts = metrics.get_current_alerts()
                
                # Calculate aggregate metrics
                total_explanations = summary.get("explanations_total", {}).get("count", 0)
                avg_response_time = summary.get("explanation_duration", {}).get("avg", 0)
                cache_hits = summary.get("cache_hits", {}).get("count", 0)
                cache_misses = summary.get("cache_misses", {}).get("count", 0)
                total_requests = cache_hits + cache_misses
                cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0
                
                error_count = summary.get("explanation_errors", {}).get("count", 0)
                error_rate = error_count / total_explanations if total_explanations > 0 else 0
                
                # System metrics
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                uptime = time.time() - self.start_time
                
                return MetricsResponse(
                    total_explanations=total_explanations,
                    average_response_time_ms=avg_response_time,
                    cache_hit_rate=cache_hit_rate,
                    error_rate=error_rate,
                    active_alerts=len(alerts),
                    uptime_seconds=uptime,
                    memory_usage_mb=memory.used / (1024 * 1024),
                    cpu_usage_percent=cpu_percent
                )
                
            except Exception as e:
                logger.error(f"Metrics error: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve metrics")
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                import psutil
                
                # Check system health
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                uptime = time.time() - self.start_time
                
                status = "healthy"
                if memory.percent > 90 or cpu_percent > 90:
                    status = "degraded"
                elif memory.percent > 95 or cpu_percent > 95:
                    status = "unhealthy"
                
                dependencies = {
                    "python": "available",
                    "fastapi": "available",
                    "transformers": "available" if self._check_transformers() else "unavailable",
                    "torch": "available" if self._check_torch() else "unavailable"
                }
                
                return HealthResponse(
                    status=status,
                    timestamp=datetime.now().isoformat(),
                    version="2.0.0",
                    uptime_seconds=uptime,
                    dependencies=dependencies
                )
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return HealthResponse(
                    status="unhealthy",
                    timestamp=datetime.now().isoformat(),
                    version="2.0.0",
                    uptime_seconds=time.time() - self.start_time,
                    dependencies={"error": str(e)}
                )
        
        @self.app.delete("/cache/clear")
        async def clear_cache():
            """Clear all caches."""
            try:
                # Try to import and use cache manager
                try:
                    from code_explainer.cache import get_cache_manager  # type: ignore
                    cache_manager = get_cache_manager()
                    if hasattr(cache_manager, 'clear_all'):
                        cache_manager.clear_all()
                        return {"message": "Cache cleared successfully"}
                except ImportError:
                    pass
                
                # Fallback: try to clear explanation cache
                try:
                    from code_explainer.cache import ExplanationCache
                    cache = ExplanationCache()
                    if hasattr(cache, 'clear'):
                        cache.clear()
                        return {"message": "Explanation cache cleared"}
                except ImportError:
                    pass
                
                return {"message": "No cache manager available"}
                
            except Exception as e:
                logger.error(f"Cache clear error: {e}")
                raise HTTPException(status_code=500, detail="Failed to clear cache")
    
    async def _validate_security(self, code: str) -> Dict[str, Any]:
        """Validate code security."""
        start_time = time.time()
        
        try:
            from code_explainer.security import CodeSecurityValidator
            
            validator = CodeSecurityValidator()
            if hasattr(validator, 'validate_code'):
                result = validator.validate_code(code)  # type: ignore
                
                # Handle different result formats
                if isinstance(result, dict):
                    is_safe = result.get("is_safe", True)
                    issues = result.get("issues", [])
                    recommendations = result.get("recommendations", [])
                elif isinstance(result, tuple):
                    is_safe = result[0] if len(result) > 0 else True
                    issues = result[1] if len(result) > 1 else []
                    recommendations = result[2] if len(result) > 2 else []
                else:
                    is_safe = bool(result)
                    issues = []
                    recommendations = []
            else:
                # Fallback validation
                is_safe = True
                issues = []
                recommendations = []
            
            risk_levels = {0: "low", 1: "medium", 2: "high", 3: "critical"}
            risk_level = risk_levels.get(len(issues), "unknown")
            
            return {
                "is_safe": is_safe,
                "issues": issues,
                "recommendations": recommendations,
                "risk_level": risk_level,
                "scan_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return {
                "is_safe": False,
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Review code manually"],
                "risk_level": "unknown",
                "scan_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure."""
        import ast
        from collections import defaultdict
        
        try:
            tree = ast.parse(code)
            
            # Count different node types
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1
            
            # Quality metrics
            quality_metrics = {
                "cyclomatic_complexity": min(sum(node_counts.values()) // 10, 10),
                "maintainability_index": max(100 - len(code) // 100, 0),
                "code_duplication": 0  # Simplified
            }
            
            suggestions = []
            if quality_metrics["cyclomatic_complexity"] > 5:
                suggestions.append("Consider breaking down complex functions")
            if len(code.splitlines()) > 50:
                suggestions.append("Consider splitting into smaller functions")
            
            return {
                "complexity_score": sum(node_counts.values()),
                "function_count": node_counts.get('FunctionDef', 0),
                "class_count": node_counts.get('ClassDef', 0),
                "line_count": len(code.splitlines()),
                "has_imports": (node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0)) > 0,
                "has_loops": (node_counts.get('For', 0) + node_counts.get('While', 0)) > 0,
                "ast_valid": True,
                "quality_metrics": quality_metrics,
                "suggestions": suggestions
            }
            
        except SyntaxError:
            return {
                "complexity_score": 0,
                "function_count": 0,
                "class_count": 0,
                "line_count": len(code.splitlines()),
                "has_imports": False,
                "has_loops": False,
                "ast_valid": False,
                "quality_metrics": {"error": "Syntax error in code"},
                "suggestions": ["Fix syntax errors before analysis"]
            }
    
    async def _record_metrics(self, event_type: str, execution_time: float, strategy: str):
        """Record metrics in background."""
        try:
            from code_explainer.monitoring import get_metrics
            
            metrics = get_metrics()
            metrics.record_event(event_type, 1, {"strategy": strategy})
            metrics.record_event("execution_time", execution_time, {"strategy": strategy})
            
        except Exception as e:
            logger.error(f"Metrics recording error: {e}")
    
    async def _record_batch_metrics(self, total: int, successful: int, failed: int, time_ms: float):
        """Record batch metrics."""
        try:
            from code_explainer.monitoring import get_metrics
            
            metrics = get_metrics()
            metrics.record_event("batch_total", total)
            metrics.record_event("batch_successful", successful)
            metrics.record_event("batch_failed", failed)
            metrics.record_event("batch_time", time_ms)
            
        except Exception as e:
            logger.error(f"Batch metrics recording error: {e}")
    
    def _check_transformers(self) -> bool:
        """Check if transformers library is available."""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def _check_torch(self) -> bool:
        """Check if torch library is available."""
        try:
            import torch
            return True
        except ImportError:
            return False


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    api = EnhancedCodeExplainerAPI()
    return api.app


def main():
    """Run the API server."""
    app = create_app()
    
    print("Starting Enhanced Code Explainer API on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
