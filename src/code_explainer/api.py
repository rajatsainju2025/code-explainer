"""FastAPI server for code explanation API."""

import logging
import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import CodeExplainer
from .utils import detect_language

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Code Explainer API",
    description="AI-powered code explanation service with multiple prompt strategies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
explainer: Optional[CodeExplainer] = None


class ExplainRequest(BaseModel):
    """Request model for code explanation."""

    code: str = Field(..., description="The code to explain", min_length=1)
    strategy: Optional[str] = Field(
        "vanilla",
        description="Prompt strategy to use",
        pattern="^(vanilla|ast_augmented|retrieval_augmented|execution_trace|enhanced_rag)$",
    )
    include_symbolic: bool = Field(False, description="Include symbolic analysis")
    use_multi_agent: bool = Field(False, description="Use multi-agent analysis")


class ExplainResponse(BaseModel):
    """Response model for code explanation."""

    explanation: str = Field(..., description="The generated explanation")
    language: str = Field(..., description="Detected programming language")
    strategy_used: str = Field(..., description="Prompt strategy that was used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AnalyzeRequest(BaseModel):
    """Request model for code analysis."""

    code: str = Field(..., description="The code to analyze", min_length=1)
    strategy: Optional[str] = Field(
        "vanilla",
        description="Prompt strategy to use",
        pattern="^(vanilla|ast_augmented|retrieval_augmented|execution_trace|enhanced_rag)$",
    )


class AnalyzeResponse(BaseModel):
    """Response model for code analysis."""

    explanation: str = Field(..., description="The generated explanation")
    line_count: int = Field(..., description="Number of lines in the code")
    character_count: int = Field(..., description="Number of characters in the code")
    language: str = Field(..., description="Detected programming language")
    contains_functions: bool = Field(..., description="Whether the code contains functions")
    contains_classes: bool = Field(..., description="Whether the code contains classes")
    contains_loops: bool = Field(..., description="Whether the code contains loops")
    contains_conditionals: bool = Field(..., description="Whether the code contains conditionals")
    contains_imports: bool = Field(..., description="Whether the code contains imports")
    strategy_used: str = Field(..., description="Prompt strategy that was used")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")


class StrategiesResponse(BaseModel):
    """Response model for available strategies."""

    strategies: List[Dict[str, str]] = Field(..., description="Available prompt strategies")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global explainer
    try:
        model_path = os.getenv("MODEL_PATH", "./results")
        config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
        explainer = CodeExplainer(model_path=model_path, config_path=config_path)
        logger.info("Model loaded successfully from %s", model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        explainer = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if explainer is not None else "unhealthy",
        version="1.0.0",
        model_loaded=explainer is not None,
    )


@app.get("/version")
async def get_version():
    """Get API version."""
    return {"version": "1.0.0", "name": "Code Explainer API"}


# Pre-cached strategies list for O(1) endpoint response
_CACHED_STRATEGIES = [
    {"name": "vanilla", "description": "Basic prompt without augmentation"},
    {"name": "ast_augmented", "description": "Enhanced with AST structure analysis"},
    {"name": "retrieval_augmented", "description": "Enhanced with docstring and context"},
    {"name": "execution_trace", "description": "Enhanced with safe execution trace"},
    {"name": "enhanced_rag", "description": "Enhanced with code similarity retrieval"},
]


@app.get("/strategies", response_model=StrategiesResponse)
async def get_strategies():
    """Get available prompt strategies."""
    return StrategiesResponse(strategies=_CACHED_STRATEGIES)


@app.post("/explain", response_model=ExplainResponse)
async def explain_code(request: ExplainRequest):
    """Explain a code snippet."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Detect language
        language = detect_language(request.code)

        # Generate explanation based on request parameters
        if request.use_multi_agent:
            explanation = explainer.explain_code_multi_agent(
                request.code, strategy=request.strategy
            )
        elif request.include_symbolic:
            explanation = explainer.explain_code_with_symbolic(
                request.code, include_symbolic=True, strategy=request.strategy
            )
        else:
            explanation = explainer.explain_code(request.code, strategy=request.strategy)

        # Use count() for line counting - more efficient than split()
        metadata = {
            "line_count": request.code.count("\n") + 1,
            "character_count": len(request.code),
            "include_symbolic": request.include_symbolic,
            "use_multi_agent": request.use_multi_agent,
        }

        return ExplainResponse(
            explanation=explanation,
            language=language,
            strategy_used=request.strategy or "vanilla",
            metadata=metadata,
        )

    except Exception as e:
        logger.error("Error explaining code: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to explain code: {str(e)}")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_code(request: AnalyzeRequest):
    """Analyze a code snippet and return detailed metrics."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get analysis from explainer
        analysis = explainer.analyze_code(request.code, strategy=request.strategy)

        return AnalyzeResponse(
            explanation=analysis["explanation"],
            line_count=analysis["line_count"],
            character_count=analysis["character_count"],
            language=analysis["language"],
            contains_functions=analysis["contains_functions"],
            contains_classes=analysis["contains_classes"],
            contains_loops=analysis["contains_loops"],
            contains_conditionals=analysis["contains_conditionals"],
            contains_imports=analysis["contains_imports"],
            strategy_used=request.strategy or "vanilla",
        )

    except Exception as e:
        logger.error("Error analyzing code: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to analyze code: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Code Explainer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "strategies": "/strategies",
    }


def main():
    """Run the API server."""
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "code_explainer.api:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
