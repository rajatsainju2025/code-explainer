from fastapi import FastAPI
from pydantic import BaseModel

from ..model import CodeExplainer

app = FastAPI(title="Code Explainer API")
explainer = CodeExplainer()


class ExplainRequest(BaseModel):
    code: str


@app.post("/explain")
async def explain(req: ExplainRequest):
    return {"explanation": explainer.explain_code(req.code)}
