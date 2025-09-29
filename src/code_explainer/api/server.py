import os
import logging
import uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # lazy fallback

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True

    # Metrics
    REQUESTS = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
    RETRIEVAL_REQUESTS = Counter('retrieval_requests_total', 'Total retrieval requests')
    RETRIEVAL_ERRORS = Counter('retrieval_errors_total', 'Total retrieval errors')
    RETRIEVAL_DURATION = Histogram('retrieval_duration_seconds', 'Retrieval request duration')

except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock metrics for when prometheus is not available
    class MockMetric:
        def inc(self): pass
        def observe(self, value=None): pass
        def start_timer(self): return self

    REQUESTS = MockMetric()
    RETRIEVAL_REQUESTS = MockMetric()
    RETRIEVAL_ERRORS = MockMetric()
    RETRIEVAL_DURATION = MockMetric()

from .. import __version__
from ..model import CodeExplainer
try:  # pragma: no cover - optional dependency
    from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
    from slowapi.errors import RateLimitExceeded  # type: ignore
    from slowapi.middleware import SlowAPIMiddleware  # type: ignore
    from slowapi.util import get_remote_address  # type: ignore
    SLOWAPI_AVAILABLE = True
except Exception:  # pragma: no cover
    SLOWAPI_AVAILABLE = False
    Limiter = None  # type: ignore
    RateLimitExceeded = Exception  # type: ignore
    SlowAPIMiddleware = None  # type: ignore
    def get_remote_address(request):  # type: ignore
        return "0.0.0.0"
    def _rate_limit_exceeded_handler(request, exc):  # type: ignore
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
from ..retrieval import CodeRetriever

# Setup logging
logger = logging.getLogger(__name__)

# Enhanced FastAPI app with comprehensive OpenAPI documentation
app = FastAPI(
    title="Code Explainer API",
    description="""
    AI-powered code explanation and retrieval service.

    This API provides intelligent code analysis capabilities including:
    * **Code Explanation**: Generate natural language explanations for Python code
    * **Retrieval-Augmented Generation**: Find similar code examples for context
    * **Multiple Strategies**: Choose from vanilla, AST-augmented, retrieval-augmented, and execution trace methods
    * **Symbolic Analysis**: Include symbolic execution information in explanations

    ## Features

    * **Multi-strategy explanation** with configurable prompt engineering
    * **Advanced code retrieval** with FAISS, BM25, and hybrid search
    * **Cross-encoder reranking** for improved relevance
    * **MMR (Maximal Marginal Relevance)** for result diversity
    * **Comprehensive metrics** and health monitoring
    * **Rate limiting** and request tracking

    ## Authentication

    Currently, this API does not require authentication for basic usage.
    Rate limiting may be applied based on server configuration.

    ## Error Handling

    The API uses standard HTTP status codes:
    * `200` - Success
    * `400` - Bad Request (invalid input)
    * `429` - Too Many Requests (rate limited)
    * `500` - Internal Server Error
    * `503` - Service Unavailable (retrieval service not ready)
    """,
    version=__version__,
    contact={
        "name": "Code Explainer Team",
        "url": "https://github.com/rsainju/code-explainer",
        "email": "code-explainer@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and monitoring endpoints",
        },
        {
            "name": "explanation",
            "description": "Code explanation and analysis",
        },
        {
            "name": "retrieval",
            "description": "Code retrieval and similarity search",
        },
        {
            "name": "metrics",
            "description": "Performance metrics and monitoring",
        },
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Request ID middleware
def get_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())

# CORS (allow local dev tools by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter configuration (optional). To disable, set CODE_EXPLAINER_RATE_LIMIT="".
RATE_LIMIT = os.environ.get("CODE_EXPLAINER_RATE_LIMIT", "60/minute")
if RATE_LIMIT and SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])  # type: ignore
    app.state.limiter = limiter  # type: ignore[attr-defined]
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore
    app.add_middleware(SlowAPIMiddleware)  # type: ignore

# Request ID and simple metrics
import time
import uuid
from typing import Any

from fastapi.responses import PlainTextResponse

# Prometheus (optional, with fallbacks safe for type-checkers)
try:  # pragma: no cover - optional dependency
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest  # type: ignore
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain"

    class _NoopMetric:
        def labels(self, **kwargs):
            return self
        def inc(self, *a, **k):  # noqa: D401
            return None
        def observe(self, *a, **k):
            return None
        def start_timer(self):  # Add start_timer method
            return self

    def Counter(*args, **kwargs):  # type: ignore
        return _NoopMetric()

    def Histogram(*args, **kwargs):  # type: ignore
        return _NoopMetric()

    def generate_latest() -> bytes:  # type: ignore
        return b""

REQUEST_COUNT: Any = Counter(
    "cx_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY: Any = Histogram(
    "cx_request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        method = request.method
        status = getattr(response, "status_code", 500)
        REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
        # Propagate request id
        if response is not None:
            response.headers["X-Request-ID"] = request_id


@app.get(
    "/metrics",
    tags=["metrics"],
    summary="Prometheus Metrics",
    description="Exposes Prometheus metrics for monitoring API performance and usage.",
    responses={
        200: {
            "description": "Prometheus metrics in text format",
            "content": {
                "text/plain": {
                    "example": "# HELP cx_requests_total Total HTTP requests\n# TYPE cx_requests_total counter\ncx_requests_total{method=\"POST\",path=\"/explain\",status=\"200\"} 42\n"
                }
            }
        }
    }
)
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

# Configurable model/config via environment
MODEL_PATH = os.environ.get("CODE_EXPLAINER_MODEL_PATH", "./results")
CONFIG_PATH = os.environ.get("CODE_EXPLAINER_CONFIG_PATH", "configs/default.yaml")
RETRIEVAL_INDEX_PATH = os.environ.get(
    "CODE_EXPLAINER_INDEX_PATH", "data/code_retrieval_index.faiss"
)
RETRIEVAL_WARMUP = os.environ.get("CODE_EXPLAINER_RETRIEVAL_WARMUP", "false").lower() in {
    "1",
    "true",
    "yes",
}
TOP_K_MAX = int(os.environ.get("CODE_EXPLAINER_RETRIEVAL_TOPK_MAX", "20"))

explainer = CodeExplainer(model_path=MODEL_PATH, config_path=CONFIG_PATH)
retriever: Optional[CodeRetriever] = None
retrieval_warmed_up: bool = False
last_retrieval_error: Optional[str] = None

# Optional warm-up on startup (loads embedding model; may be heavy)
if RETRIEVAL_WARMUP and os.path.exists(RETRIEVAL_INDEX_PATH):
    try:
        retriever = CodeRetriever()
        retriever.load_index(RETRIEVAL_INDEX_PATH)
        retrieval_warmed_up = True
    except Exception as e:  # pragma: no cover
        last_retrieval_error = f"Warm-up failed: {e}"


class ExplainRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python code to explain")
    # Allowed strategies: vanilla | ast_augmented | retrieval_augmented | execution_trace | enhanced_rag
    strategy: Optional[str] = Field(default=None, description="Prompt strategy to use")
    symbolic: Optional[bool] = Field(default=False, description="Include symbolic analysis")


class RetrievalRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=200000,
        description="Code snippet to search for similar examples",
        alias="code"
    )
    k: int = Field(
        default=3, ge=1, le=50, description="Number of similar examples to retrieve", alias="top_k"
    )
    method: str = Field(default="hybrid", description="Retrieval method: faiss|bm25|hybrid")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid weight toward FAISS similarity")
    use_reranker: bool = Field(default=False, description="Use cross-encoder reranking")
    use_mmr: bool = Field(default=False, description="Use MMR for diversity")
    rerank_top_k: int = Field(default=20, ge=1, le=100, description="Candidates to rerank")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0, description="MMR relevance-diversity trade-off")

    class Config:
        allow_population_by_field_name = True


class RetrievalResponse(BaseModel):
    similar_codes: List[str]
    metadata: Dict[str, Any]


class EnhancedRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=200000, description="Code query")
    k: int = Field(default=3, ge=1, le=50, description="Number of results")
    method: str = Field(default="hybrid", description="Retrieval method")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid weight")
    use_reranker: bool = Field(default=True, description="Use cross-encoder reranking")
    use_mmr: bool = Field(default=True, description="Use MMR for diversity")
    rerank_top_k: int = Field(default=20, ge=1, le=100, description="Candidates to rerank")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0, description="MMR trade-off")


class EnhancedRetrievalResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    method_used: str
    total_results: int


@app.get(
    "/health",
    tags=["health"],
    summary="Health Check",
    description="Check the overall health of the API service and its components.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "retrieval": {
                            "index_path": "data/code_retrieval_index.faiss",
                            "index_exists": True,
                            "corpus_exists": True,
                            "index_size": 1000,
                            "warmed_up": True,
                            "last_error": None,
                            "top_k_max": 20
                        }
                    }
                }
            }
        }
    }
)
async def health():
    # Retrieval quick status (without loading model)
    index_exists = os.path.exists(RETRIEVAL_INDEX_PATH)
    corpus_exists = os.path.exists(f"{RETRIEVAL_INDEX_PATH}.corpus.json")
    index_size = None
    if faiss and index_exists:
        try:
            # Read header cheaply to get size
            idx = faiss.read_index(RETRIEVAL_INDEX_PATH)
            index_size = getattr(idx, "ntotal", None)
        except Exception:
            index_size = None

    return {
        "status": "ok",
        "retrieval": {
            "index_path": RETRIEVAL_INDEX_PATH,
            "index_exists": index_exists,
            "corpus_exists": corpus_exists,
            "index_size": index_size,
            "warmed_up": retrieval_warmed_up,
            "last_error": last_retrieval_error,
            "top_k_max": TOP_K_MAX,
        },
    }


@app.get(
    "/version",
    tags=["health"],
    summary="Get API Version",
    description="Returns the current version of the Code Explainer API.",
    responses={
        200: {
            "description": "Version information",
            "content": {
                "application/json": {
                    "example": {"version": "0.3.0"}
                }
            }
        }
    }
)
async def version():
    return {"version": __version__}


@app.post(
    "/explain",
    tags=["explanation"],
    summary="Explain Python Code",
    description="""
    Generate a natural language explanation for the provided Python code.

    This endpoint supports multiple explanation strategies:
    * **vanilla**: Basic prompt engineering
    * **ast_augmented**: Includes AST analysis for better understanding
    * **retrieval_augmented**: Uses similar code examples for context
    * **execution_trace**: Includes execution path analysis

    The `symbolic` parameter enables symbolic execution analysis for more detailed explanations.
    """,
    responses={
        200: {
            "description": "Code explanation generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "explanation": "This function calculates the factorial of a number using recursion. It takes an integer n as input and returns n! (n factorial). The base case is when n is 0 or 1, returning 1. Otherwise, it multiplies n by the factorial of n-1."
                    }
                }
            }
        },
        400: {
            "description": "Invalid request - empty or invalid code",
            "content": {
                "application/json": {
                    "example": {"detail": "code must not be empty"}
                }
            }
        },
        500: {
            "description": "Internal server error during explanation generation"
        }
    }
)
async def explain(req: ExplainRequest):
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="code must not be empty")

    if req.symbolic:
        explanation = explainer.explain_code_with_symbolic(
            req.code, include_symbolic=True, strategy=req.strategy
        )
    else:
        explanation = explainer.explain_code(req.code, strategy=req.strategy)
    return {"explanation": explanation}


@app.get(
    "/strategies",
    tags=["explanation"],
    summary="List Available Strategies",
    description="Returns a list of all available explanation strategies supported by the API.",
    responses={
        200: {
            "description": "List of available strategies",
            "content": {
                "application/json": {
                    "example": {
                        "strategies": [
                            "vanilla",
                            "ast_augmented",
                            "retrieval_augmented",
                            "execution_trace"
                        ]
                    }
                }
            }
        }
    }
)
async def strategies():
    return {
        "strategies": [
            "vanilla",
            "ast_augmented",
            "retrieval_augmented",
            "execution_trace",
        ]
    }


@app.post(
    "/retrieve",
    tags=["retrieval"],
    summary="Retrieve Similar Code",
    description="""
    Find similar code snippets using various retrieval methods.

    Supports multiple retrieval strategies:
    * **faiss**: Pure vector similarity search using FAISS
    * **bm25**: Text-based retrieval using BM25 algorithm
    * **hybrid**: Combination of vector and text-based search

    Optional advanced features:
    * **Cross-encoder reranking** for improved relevance
    * **MMR (Maximal Marginal Relevance)** for result diversity
    """,
    responses={
        200: {
            "description": "Similar code snippets retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "similar_codes": [
                            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
                        ],
                        "metadata": {
                            "enhanced": True,
                            "reranker_used": True,
                            "mmr_used": False,
                            "results_metadata": [...]
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters"
        },
        503: {
            "description": "Retrieval service not available",
            "content": {
                "application/json": {
                    "example": {"detail": "Retrieval service not available"}
                }
            }
        }
    }
)
async def retrieve_similar(
    request: RetrievalRequest,
    request_id: str = Depends(get_request_id)
) -> RetrievalResponse:
    """Retrieve similar code snippets with enhanced features."""
    try:
        # Update metrics
        RETRIEVAL_REQUESTS.inc()
        retrieval_timer = RETRIEVAL_DURATION.start_timer()

        logger.info(f"Retrieval request: method={request.method}, k={request.k}, request_id={request_id}")

        if retriever is None:
            raise HTTPException(status_code=503, detail="Retrieval service not available")

        # Use enhanced retrieval if advanced features are requested
        if request.use_reranker or request.use_mmr:
            results = retriever.retrieve_similar_code_enhanced(
                query_code=request.query,
                k=request.k,
                method=request.method,
                alpha=request.alpha,
                use_reranker=request.use_reranker,
                use_mmr=request.use_mmr,
                rerank_top_k=request.rerank_top_k,
                mmr_lambda=request.mmr_lambda
            )
            # Convert enhanced results to simple format for compatibility
            similar_codes = [result["content"] for result in results]
            metadata = {
                "enhanced": True,
                "reranker_used": request.use_reranker,
                "mmr_used": request.use_mmr,
                "results_metadata": results
            }
        else:
            # Use simple retrieval
            similar_codes = retriever.retrieve_similar_code(
                query_code=request.query,
                k=request.k,
                method=request.method,
                alpha=request.alpha
            )
            metadata = {"enhanced": False}

        retrieval_timer.observe()

        return RetrievalResponse(
            similar_codes=similar_codes,
            metadata=metadata
        )

    except Exception as e:
        RETRIEVAL_ERRORS.inc()
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.post(
    "/retrieve/enhanced",
    tags=["retrieval"],
    summary="Enhanced Code Retrieval",
    description="""
    Advanced code retrieval with reranking and diversity features.

    This endpoint provides enhanced retrieval capabilities with:
    * **Cross-encoder reranking** for better relevance scoring
    * **MMR (Maximal Marginal Relevance)** to balance relevance and diversity
    * **Detailed metadata** for each result including scores and rankings
    """,
    responses={
        200: {
            "description": "Enhanced retrieval completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "content": "def factorial(n): ...",
                                "score": 0.95,
                                "rerank_score": 0.87,
                                "metadata": {...}
                            }
                        ],
                        "query": "recursive function",
                        "method_used": "hybrid",
                        "total_results": 3
                    }
                }
            }
        },
        503: {
            "description": "Retrieval service not available"
        }
    }
)
async def retrieve_similar_enhanced(
    request: EnhancedRetrievalRequest,
    request_id: str = Depends(get_request_id)
) -> EnhancedRetrievalResponse:
    """Enhanced retrieval with reranking and MMR for diversity."""
    try:
        # Update metrics
        RETRIEVAL_REQUESTS.inc()
        retrieval_timer = RETRIEVAL_DURATION.start_timer()

        logger.info(f"Enhanced retrieval: method={request.method}, rerank={request.use_reranker}, mmr={request.use_mmr}, request_id={request_id}")

        if retriever is None:
            raise HTTPException(status_code=503, detail="Retrieval service not available")

        results = retriever.retrieve_similar_code_enhanced(
            query_code=request.query,
            k=request.k,
            method=request.method,
            alpha=request.alpha,
            use_reranker=request.use_reranker,
            use_mmr=request.use_mmr,
            rerank_top_k=request.rerank_top_k,
            mmr_lambda=request.mmr_lambda
        )

        retrieval_timer.observe()

        return EnhancedRetrievalResponse(
            results=results,
            query=request.query,
            method_used=request.method,
            total_results=len(results)
        )

    except Exception as e:
        RETRIEVAL_ERRORS.inc()
        logger.error(f"Enhanced retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced retrieval failed: {str(e)}")


@app.get(
    "/retrieval/health",
    tags=["health"],
    summary="Retrieval Service Health",
    description="Detailed health check for the code retrieval service without loading models.",
    responses={
        200: {
            "description": "Retrieval service health information",
            "content": {
                "application/json": {
                    "example": {
                        "index_path": "data/code_retrieval_index.faiss",
                        "index_exists": True,
                        "corpus_exists": True,
                        "index_size": 1000,
                        "warmed_up": True,
                        "last_error": None,
                        "top_k_max": 20
                    }
                }
            }
        }
    }
)
async def retrieval_health():
    """Detailed retrieval health without requiring model load."""
    index_exists = os.path.exists(RETRIEVAL_INDEX_PATH)
    corpus_exists = os.path.exists(f"{RETRIEVAL_INDEX_PATH}.corpus.json")
    index_size = None
    error = None

    if faiss and index_exists:
        try:
            idx = faiss.read_index(RETRIEVAL_INDEX_PATH)
            index_size = getattr(idx, "ntotal", None)
        except Exception as e:  # pragma: no cover
            error = str(e)

    return {
        "index_path": RETRIEVAL_INDEX_PATH,
        "index_exists": index_exists,
        "corpus_exists": corpus_exists,
        "index_size": index_size,
        "warmed_up": retrieval_warmed_up,
        "last_error": error or last_retrieval_error,
        "top_k_max": TOP_K_MAX,
    }


@app.get(
    "/retrieval/stats",
    tags=["metrics"],
    summary="Retrieval Statistics",
    description="Runtime statistics and information about the loaded retrieval service.",
    responses={
        200: {
            "description": "Retrieval service statistics",
            "content": {
                "application/json": {
                    "example": {
                        "loaded": True,
                        "index_loaded": True,
                        "corpus_size": 1000,
                        "faiss_ntotal": 1000,
                        "bm25_built": True,
                        "index_path": "data/code_retrieval_index.faiss",
                        "top_k_max": 20
                    }
                }
            }
        }
    }
)
async def retrieval_stats():
    """Runtime stats of the loaded retriever (if any)."""
    info = {
        "loaded": retriever is not None,
        "index_loaded": bool(getattr(retriever, "index", None)),
        "corpus_size": len(getattr(retriever, "code_corpus", []) or []),
        "faiss_ntotal": getattr(getattr(retriever, "index", None), "ntotal", None),
        "bm25_built": bool(getattr(retriever, "_bm25", None)),
        "index_path": RETRIEVAL_INDEX_PATH,
        "top_k_max": TOP_K_MAX,
    }
    return info
