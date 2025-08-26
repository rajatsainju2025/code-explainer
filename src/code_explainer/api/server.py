import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
explainer = CodeExplainer(model_path=MODEL_PATH, config_path=CONFIG_PATH)
retriever: Optional[CodeRetriever] = None


class ExplainRequest(BaseModel):
    code: str
    # Allowed strategies: vanilla | ast_augmented | retrieval_augmented | execution_trace
    strategy: Optional[str] = None
    symbolic: Optional[bool] = False


class RetrieveRequest(BaseModel):
    code: str
    index_path: str = "data/code_retrieval_index.faiss"
    top_k: int = 3


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {"version": __version__}


@app.post("/explain")
async def explain(req: ExplainRequest):
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
    if retriever is None:
        retriever = CodeRetriever()
    # Load index on demand
    try:
        retriever.load_index(req.index_path)
    except FileNotFoundError:
        return {"error": f"Index not found: {req.index_path}"}
    except Exception as e:
        return {"error": f"Failed to load index: {e}"}

    try:
        matches = retriever.retrieve_similar_code(req.code, k=req.top_k)
        return {"matches": matches, "count": len(matches)}
    except Exception as e:
        return {"error": f"Failed to retrieve: {e}"}
