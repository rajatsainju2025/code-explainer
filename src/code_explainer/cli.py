"""Command line interface for code explainer.

Aliases (installed via console scripts):
- cx-train -> code_explainer.cli:train
- cx-serve -> code_explainer.cli:serve
- cx-explain -> code_explainer.cli:explain
- cx-explain-file -> code_explainer.cli:explain_file

Examples:
  cx-train --config configs/default.yaml
  cx-explain "print('hello')"
  cx-explain-file path/to/script.py
  cx-serve --host 0.0.0.0 --port 7860
"""

import click
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .trainer import CodeExplainerTrainer
from .model import CodeExplainer
from .utils import setup_logging

console = Console()


@click.group(help="Code Explainer CLI - Train and use LLM models for code explanation.\n\nAliases: cx-train, cx-serve, cx-explain, cx-explain-file")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose):
    """Code Explainer CLI - Train and use LLM models for code explanation."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)


@main.command()
@click.option('--config', '-c', default='configs/default.yaml', 
              help='Path to configuration file')
@click.option('--data', '-d', help='Path to training data (JSON format)')
def train(config, data):
    """Train a new code explanation model."""
    console.print(Panel.fit("🚀 Starting Model Training", style="bold blue"))
    
    try:
        trainer = CodeExplainerTrainer(config_path=config)
        trainer.train(data_path=data)
        console.print(Panel.fit("✅ Training completed successfully!", style="bold green"))
    except Exception as e:
        console.print(Panel.fit(f"❌ Training failed: {e}", style="bold red"))
        raise


@main.command()
@click.option('--model-path', '-m', default='./results', 
              help='Path to trained model')
@click.option('--config', '-c', default='configs/default.yaml',
              help='Path to configuration file')
@click.argument('code', required=False)
def explain(model_path, config, code):
    """Explain a code snippet."""
    explainer = CodeExplainer(model_path=model_path, config_path=config)
    
    if code is None:
        # Interactive mode
        console.print(Panel.fit("🐍 Interactive Code Explanation Mode", style="bold blue"))
        console.print("Enter Python code (press Ctrl+D to finish, Ctrl+C to exit):")
        
        while True:
            try:
                lines = []
                while True:
                    try:
                        line = input(">>> " if not lines else "... ")
                        lines.append(line)
                    except EOFError:
                        break
                
                if not lines:
                    continue
                    
                code = '\n'.join(lines)
                
                # Display code with syntax highlighting
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title="Code", border_style="blue"))
                
                # Generate explanation
                with console.status("[bold green]Generating explanation..."):
                    explanation = explainer.explain_code(code)
                
                console.print(Panel(explanation, title="Explanation", border_style="green"))
                console.print("-" * 50)
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
    else:
        # Single explanation mode
        explanation = explainer.explain_code(code)
        
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Code", border_style="blue"))
        console.print(Panel(explanation, title="Explanation", border_style="green"))


@main.command()
@click.option('--model-path', '-m', default='./results',
              help='Path to trained model')
@click.option('--config', '-c', default='configs/default.yaml',
              help='Path to configuration file')
@click.argument('file_path', type=click.Path(exists=True))
def explain_file(model_path, config, file_path):
    """Explain code from a Python file."""
    explainer = CodeExplainer(model_path=model_path, config_path=config)
    
    with open(file_path, 'r') as f:
        code = f.read()
    
    console.print(f"📁 Explaining file: {file_path}")
    
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Code", border_style="blue"))
    
    with console.status("[bold green]Generating explanation..."):
        analysis = explainer.analyze_code(code)
    
    console.print(Panel(analysis["explanation"], title="Explanation", border_style="green"))
    
    # Show analysis
    analysis_text = f"""
📊 **Code Analysis:**
- Lines: {analysis['line_count']}
- Characters: {analysis['character_count']}
- Contains functions: {'✅' if analysis['contains_functions'] else '❌'}
- Contains classes: {'✅' if analysis['contains_classes'] else '❌'}
- Contains loops: {'✅' if analysis['contains_loops'] else '❌'}
- Contains conditionals: {'✅' if analysis['contains_conditionals'] else '❌'}
- Contains imports: {'✅' if analysis['contains_imports'] else '❌'}
    """
    console.print(Panel(analysis_text, title="Analysis", border_style="yellow"))


@main.command()
@click.option('--host', default='127.0.0.1', help='Host address')
@click.option('--port', default=7860, help='Port number')
@click.option('--model-path', '-m', default='./results',
              help='Path to trained model')
def serve(host, port, model_path):
    """Start the web interface."""
    console.print(Panel.fit("🌐 Starting Web Interface", style="bold blue"))
    
    try:
        # Import here to avoid dependency issues
        import gradio as gr
        from .model import CodeExplainer
        
        explainer = CodeExplainer(model_path=model_path)
        
        def explain_code_web(code_snippet):
            if not code_snippet.strip():
                return "Please enter some Python code to explain."
            return explainer.explain_code(code_snippet)
        
        # Create Gradio interface
        iface = gr.Interface(
            fn=explain_code_web,
            inputs=gr.Code(language="python", label="Python Code"),
            outputs=gr.Textbox(label='Explanation'),
            title="🐍 Python Code Explainer",
            description="Enter a snippet of Python code and get an AI-generated explanation.",
            flagging_mode="never",
            examples=[
                ["def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"],
                ["with open('file.txt', 'r') as f:\n    content = f.read()"],
                ["squares = [x**2 for x in range(10)]"],
            ]
        )
        
        console.print(f"🚀 Server starting at http://{host}:{port}")
        iface.launch(server_name=host, server_port=port)
        
    except ImportError:
        console.print(Panel.fit("❌ Gradio not installed. Install with: pip install gradio", 
                               style="bold red"))
    except Exception as e:
        console.print(Panel.fit(f"❌ Failed to start server: {e}", style="bold red"))


@main.command()
@click.option('--model-path', '-m', default='./results', help='Path to trained model directory')
@click.option('--config', '-c', default='configs/default.yaml', help='Path to configuration file')
@click.option('--test-file', '-t', default=None, help='Optional path to test JSON overriding config')
def eval(model_path, config, test_file):
    """Evaluate a model on a test set (BLEU/ROUGE/BERTScore)."""
    from .model import CodeExplainer
    from .metrics.evaluate import compute_bleu, compute_rouge_l, compute_codebert_score
    import json

    console.print(Panel.fit("📏 Running evaluation", style="bold blue"))
    explainer = CodeExplainer(model_path=model_path, config_path=config)

    # Load test data
    if test_file is None:
        cfg = explainer.config
        test_file = cfg.get('data', {}).get('test_file')
    if not test_file:
        console.print(Panel.fit("❌ No test file provided or configured.", style="bold red"))
        return
    with open(test_file, 'r') as f:
        data = json.load(f)

    refs = []
    preds = []
    for ex in data:
        code = ex.get('code', '')
        ref = ex.get('explanation', '')
        try:
            pred = explainer.explain_code(code)
        except Exception:
            pred = ""
        refs.append(ref)
        preds.append(pred)

    bleu = compute_bleu(refs, preds)
    rougeL = compute_rouge_l(refs, preds)
    bert = compute_codebert_score(refs, preds)

    metrics = {"bleu": bleu, "rougeL": rougeL, "bert_score": bert}
    console.print(Panel.fit(str(metrics), style="bold green"))


if __name__ == "__main__":
    main()
