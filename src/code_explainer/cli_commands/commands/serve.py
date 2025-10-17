"""Serve commands for CLI."""

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


def register_serve_commands(main_group):
    """Register serving-related commands."""

    @main_group.command()
    @click.option("--host", default="127.0.0.1", help="Host address")
    @click.option("--port", default=7860, help="Port number")
    @click.option("--model-path", "-m", default="./results", help="Path to trained model")
    def serve(host, port, model_path):
        """Start the web interface."""
        console.print(Panel.fit("🌐 Starting Web Interface", style="bold blue"))

        try:
            # Import here to avoid dependency issues
            import gradio as gr

            from ...model import CodeExplainer

            explainer = CodeExplainer(model_path=model_path)

            def explain_code_web(code_snippet, strategy="vanilla"):
                if not code_snippet.strip():
                    return "Please enter some Python code to explain."
                return explainer.explain_code(code_snippet, strategy=strategy)

            # Create Gradio interface
            iface = gr.Interface(
                fn=explain_code_web,
                inputs=[
                    gr.Textbox(lines=10, placeholder="Enter Python code here..."),
                    gr.Radio(["vanilla", "ast_augmented", "retrieval_augmented"], value="vanilla")
                ],
                outputs=gr.Textbox(lines=10),
                title="Code Explainer",
                description="Explain Python code using advanced LLM techniques"
            )

            console.print(f"🚀 Server starting at http://{host}:{port}")
            iface.launch(server_name=host, server_port=port)

        except ImportError:
            console.print(Panel.fit("❌ Gradio not installed. Install with: pip install gradio", style="bold red"))
            raise
        except Exception as e:
            console.print(Panel.fit(f"❌ Failed to start server: {e}", style="bold red"))
            raise