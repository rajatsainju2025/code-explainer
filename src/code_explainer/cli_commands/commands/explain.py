"""Explanation commands for CLI."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ...model import CodeExplainer

console = Console()


def register_explain_commands(main_group):
    """Register explanation-related commands."""

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
    def explain(model_path, config, prompt_strategy, symbolic, multi_agent, code):
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

                    code = "\n".join(lines)

                    # Display code with syntax highlighting
                    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, title="Code", border_style="blue"))

                    # Generate explanation
                    with console.status("[bold green]Generating explanation..."):
                        if multi_agent:
                            explanation = explainer.explain_code_multi_agent(
                                code, strategy=prompt_strategy
                            )
                        elif symbolic:
                            explanation = explainer.explain_code_with_symbolic(
                                code, include_symbolic=True, strategy=prompt_strategy
                            )
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
                explanation = explainer.explain_code_multi_agent(code, strategy=prompt_strategy)
            elif symbolic:
                explanation = explainer.explain_code_with_symbolic(
                    code, include_symbolic=True, strategy=prompt_strategy
                )
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
        explainer = CodeExplainer(model_path=model_path, config_path=config)

        with open(file_path, "r") as f:
            code = f.read()

        console.print(f"📁 Explaining file: {file_path}")

        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Code", border_style="blue"))

        with console.status("[bold green]Generating explanation..."):
            analysis = explainer.analyze_code(code, strategy=prompt_strategy)

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