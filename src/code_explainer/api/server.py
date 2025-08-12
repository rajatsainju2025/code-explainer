from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import os

from .. import __version__
from ..model import CodeExplainer

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


class ExplainRequest(BaseModel):
    code: str
    # Allowed strategies: vanilla | ast_augmented | retrieval_augmented | execution_trace
    strategy: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {"version": __version__}


@app.post("/explain")
async def explain(req: ExplainRequest):
    return {"explanation": explainer.explain_code(req.code, strategy=req.strategy)}
