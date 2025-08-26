import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # lazy fallback

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

app = FastAPI(title="Code Explainer API")

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


@app.get("/metrics")
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


class RetrieveRequest(BaseModel):
    code: str = Field(
        ...,
        min_length=3,
        max_length=200000,
        description="Code snippet to search for similar examples",
    )
    index_path: str = Field(
        default=RETRIEVAL_INDEX_PATH, description="Path to FAISS index (built via CLI build-index)"
    )
    top_k: int = Field(
        default=3, ge=1, le=TOP_K_MAX, description="Number of similar examples to retrieve"
    )
    method: Optional[str] = Field(default="hybrid", description="Retrieval method: faiss|bm25|hybrid")
    alpha: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid weight toward FAISS similarity")


@app.get("/health")
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


@app.get("/version")
async def version():
    return {"version": __version__}


@app.post("/explain")
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


@app.get("/strategies")
async def strategies():
    return {
        "strategies": [
            "vanilla",
            "ast_augmented",
            "retrieval_augmented",
            "execution_trace",
        ]
    }


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    global retriever
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="code must not be empty")
    if req.top_k < 1 or req.top_k > TOP_K_MAX:
        raise HTTPException(status_code=400, detail=f"top_k must be between 1 and {TOP_K_MAX}")

    # Lazy-init retriever and load index on demand
    try:
        if retriever is None:
            retriever = CodeRetriever()
        retriever.load_index(req.index_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index not found: {req.index_path}")
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Dependency missing for retrieval: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index: {e}")

    try:
        matches = retriever.retrieve_similar_code(
            req.code,
            k=req.top_k,
            method=(req.method or "hybrid"),
            alpha=(req.alpha or 0.5),
        )
        return {
            "matches": matches,
            "count": len(matches),
            "method": req.method or "hybrid",
            "alpha": req.alpha or 0.5,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve: {e}")


@app.get("/retrieval/health")
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


@app.get("/retrieval/stats")
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
