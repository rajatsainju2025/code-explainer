from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .. import __version__
from ..model import CodeExplainer

app = FastAPI(title="Code Explainer API")
explainer = CodeExplainer()


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
