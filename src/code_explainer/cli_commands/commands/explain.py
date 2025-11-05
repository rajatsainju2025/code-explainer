"""Explanation commands for CLI."""

from functools import lru_cache
from typing import Any, Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ...model import CodeExplainer

console: Console = Console()


@lru_cache(maxsize=8)
def _get_explainer(model_path: str, config: str) -> CodeExplainer:
    """Cache CodeExplainer instances to avoid repeated initialization."""
    return CodeExplainer(model_path=model_path, config_path=config)


def register_explain_commands(main_group: Any) -> None:
    """Register explanation-related commands.
    
    Args:
        main_group: Click command group to register commands to
    """

    @main_group.command()
    @click.option("--model-path", "-m", default="./results", help="Path to trained model")
    @click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
    @click.option(
        "--prompt-strategy",
        type=click.Choice(
            ["vanilla", "ast_augmented", "retrieval_augmented", "execution_trace", "enhanced_rag"]
        ),
        default=None,
        help="Override prompt strategy (default from config)",
    )
    @click.option("--symbolic", is_flag=True, help="Include symbolic analysis in explanation")
    @click.option("--multi-agent", is_flag=True, help="Use multi-agent collaborative explanation")
    @click.argument("code", required=False)
    def explain(
        model_path: str,
        config: str,
        prompt_strategy: Optional[str],
        symbolic: bool,
        multi_agent: bool,
        code: Optional[str]
    ) -> None:
        """Explain a code snippet."""
        explainer: CodeExplainer = _get_explainer(model_path, config)

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

                    code = "\n".join(lines)

                    # Display code with syntax highlighting
                    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, title="Code", border_style="blue"))

                    # Generate explanation
                    with console.status("[bold green]Generating explanation..."):
                        if multi_agent:
                            explanation = explainer.multi_agent_orchestrator.explain_code_collaborative(code)
                        elif symbolic:
                            symbolic_result = explainer.symbolic_analyzer.analyze_code(code)
                            explanation = f"{explainer.explain_code(code, strategy=prompt_strategy)}\n\nSymbolic Analysis:\n{symbolic_result}"
                        else:
                            explanation = explainer.explain_code(code, strategy=prompt_strategy)

                    console.print(Panel(explanation, title="Explanation", border_style="green"))
                    console.print("-" * 50)

                except KeyboardInterrupt:
                    console.print("\n👋 Goodbye!")
                    break
        else:
            # Single explanation mode
            if multi_agent:
                explanation = explainer.multi_agent_orchestrator.explain_code_collaborative(code)
            elif symbolic:
                symbolic_result = explainer.symbolic_analyzer.analyze_code(code)
                explanation = f"{explainer.explain_code(code, strategy=prompt_strategy)}\n\nSymbolic Analysis:\n{symbolic_result}"
            else:
                explanation = explainer.explain_code(code, strategy=prompt_strategy)

            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Code", border_style="blue"))
            console.print(Panel(explanation, title="Explanation", border_style="green"))

    @main_group.command()
    @click.option("--model-path", "-m", default="./results", help="Path to trained model")
    @click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
    @click.option(
        "--prompt-strategy",
        type=click.Choice(
            ["vanilla", "ast_augmented", "retrieval_augmented", "execution_trace", "enhanced_rag"]
        ),
        default=None,
        help="Override prompt strategy (default from config)",
    )
    @click.argument("file_path", type=click.Path(exists=True))
    def explain_file(model_path, config, prompt_strategy, file_path):
        """Explain code from a Python file."""
        explainer = _get_explainer(model_path, config)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        console.print(f"📁 Explaining file: {file_path}")

        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Code", border_style="blue"))

        with console.status("[bold green]Generating explanation..."):
            explanation = explainer.explain_code(code, strategy=prompt_strategy)

        console.print(Panel(explanation, title="Explanation", border_style="green"))

        # Show basic analysis
        lines = code.split('\n')
        analysis_text = f"""
📊 **Code Analysis:**
- Lines: {len(lines)}
- Characters: {len(code)}
- Contains functions: {'def ' in code}
- Contains classes: {'class ' in code}
- Contains loops: {'for ' in code or 'while ' in code}
- Contains conditionals: {'if ' in code}
- Contains imports: {'import ' in code}
        """
        console.print(Panel(analysis_text, title="Analysis", border_style="yellow"))