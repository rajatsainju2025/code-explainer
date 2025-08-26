import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # lazy fallback

from .. import __version__
from ..model import CodeExplainer
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
        matches = retriever.retrieve_similar_code(req.code, k=req.top_k)
        return {"matches": matches, "count": len(matches)}
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
