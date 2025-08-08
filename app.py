import gradio as gr
from pathlib import Path

# Allow running from repo root by importing from src/
import sys
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_explainer.model import CodeExplainer  # noqa: E402

# Initialize explainer with defaults (will use configs/default.yaml)
explainer = CodeExplainer(model_path="./results", config_path="configs/default.yaml")


def explain_code(code_snippet: str) -> str:
    if not code_snippet.strip():
        return "Please enter some code to explain."
    return explainer.explain_code(code_snippet)


iface = gr.Interface(
    fn=explain_code,
    inputs=gr.Code(language="python", label="Code"),
    outputs=gr.Textbox(label='Explanation'),
    title="💡 Code Explainer",
    description="Enter a code snippet and get an AI-generated explanation.",
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch()