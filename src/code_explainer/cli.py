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

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .model import CodeExplainer
from .trainer import CodeExplainerTrainer
from .utils import setup_logging

console = Console()


@click.group(
    help="Code Explainer CLI - Train and use LLM models for code explanation.\n\nAliases: cx-train, cx-serve, cx-explain, cx-explain-file"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose):
    """Code Explainer CLI - Train and use LLM models for code explanation."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)


@main.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
@click.option("--data", "-d", help="Path to training data (JSON format)")
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


@main.command()
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


@main.command()
@click.option("--host", default="127.0.0.1", help="Host address")
@click.option("--port", default=7860, help="Port number")
@click.option("--model-path", "-m", default="./results", help="Path to trained model")
def serve(host, port, model_path):
    """Start the web interface."""
    console.print(Panel.fit("🌐 Starting Web Interface", style="bold blue"))

    try:
        # Import here to avoid dependency issues
        import gradio as gr

        from .model import CodeExplainer

        explainer = CodeExplainer(model_path=model_path)

        def explain_code_web(code_snippet, strategy="vanilla"):
            if not code_snippet.strip():
                return "Please enter some Python code to explain."
            return explainer.explain_code(code_snippet, strategy=strategy)

        # Create Gradio interface
        iface = gr.Interface(
            fn=explain_code_web,
            inputs=[
                gr.Code(language="python", label="Python Code"),
                gr.Dropdown(
                    choices=[
                        "vanilla",
                        "ast_augmented",
                        "retrieval_augmented",
                        "execution_trace",
                        "enhanced_rag",
                    ],
                    value="vanilla",
                    label="Prompt Strategy",
                ),
            ],
            outputs=gr.Textbox(label="Explanation"),
            title="🐍 Python Code Explainer",
            description="Enter a snippet of Python code and get an AI-generated explanation using different prompt strategies.",
            flagging_mode="never",
            examples=[
                [
                    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                    "vanilla",
                ],
                ["with open('file.txt', 'r') as f:\n    content = f.read()", "ast_augmented"],
                ["squares = [x**2 for x in range(10)]", "enhanced_rag"],
            ],
        )

        console.print(f"🚀 Server starting at http://{host}:{port}")
        iface.launch(server_name=host, server_port=port)

    except ImportError:
        console.print(
            Panel.fit("❌ Gradio not installed. Install with: pip install gradio", style="bold red")
        )
    except Exception as e:
        console.print(Panel.fit(f"❌ Failed to start server: {e}", style="bold red"))


@main.command()
@click.option("--model-path", "-m", default="./results", help="Path to trained model directory")
@click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
@click.option(
    "--test-file", "-t", default=None, help="Optional path to test JSON overriding config"
)
@click.option("--preds-out", "-o", default=None, help="Optional path to save predictions as JSONL")
@click.option(
    "--max-samples", type=int, default=None, help="Limit number of test samples (fast CI)"
)
@click.option(
    "--prompt-strategy",
    type=click.Choice(
        ["vanilla", "ast_augmented", "retrieval_augmented", "execution_trace", "enhanced_rag"]
    ),
    default=None,
    help="Override prompt strategy (default from config)",
)
def eval(model_path, config, test_file, preds_out, max_samples, prompt_strategy):
    """Evaluate a model on a test set (BLEU/ROUGE/BERTScore) and optionally save predictions."""
    import json

    from .metrics.evaluate import (
        compute_bleu,
        compute_codebert_score,
        compute_codebleu,
        compute_rouge_l,
    )
    from .model import CodeExplainer

    console.print(Panel.fit("📏 Running evaluation", style="bold blue"))
    explainer = CodeExplainer(model_path=model_path, config_path=config)

    # Load test data
    if test_file is None:
        cfg = explainer.config
        test_file = cfg.get("data", {}).get("test_file")
    if not test_file:
        console.print(Panel.fit("❌ No test file provided or configured.", style="bold red"))
        return
    with open(test_file, "r") as f:
        data = json.load(f)

    if max_samples is not None:
        try:
            k = int(max_samples)
            data = data[: max(0, k)]
        except Exception:
            pass

    refs = []
    preds = []
    codes = []
    for ex in data:
        code = ex.get("code", "")
        ref = ex.get("explanation", "")
        try:
            pred = explainer.explain_code(code, strategy=prompt_strategy)
        except Exception:
            pred = ""
        codes.append(code)
        refs.append(ref)
        preds.append(pred)

    bleu = compute_bleu(refs, preds)
    rougeL = compute_rouge_l(refs, preds)
    bert = compute_codebert_score(refs, preds)
    codebleu = compute_codebleu(refs, preds)

    # Basic confusion/quality summary
    total = len(refs) if refs else 1
    exact = sum(1 for r, p in zip(refs, preds) if p.strip() == r.strip())
    empty = sum(1 for p in preds if not p.strip())
    avg_pred_len = sum(len(p) for p in preds) / total
    avg_ref_len = sum(len(r) for r in refs) / total

    table = Table(title="Evaluation Summary")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="white")
    table.add_row("BLEU", f"{bleu:.4f}")
    table.add_row("ROUGE-L", f"{rougeL:.4f}")
    table.add_row("BERTScore", f"{bert:.4f}")
    table.add_row("CodeBLEU", f"{codebleu:.4f}")
    table.add_row("Exact match %", f"{(exact/total)*100:.2f}%")
    table.add_row("Empty preds %", f"{(empty/total)*100:.2f}%")
    table.add_row("Avg pred len", f"{avg_pred_len:.1f}")
    table.add_row("Avg ref len", f"{avg_ref_len:.1f}")
    console.print(table)

    # Optional: save predictions as JSONL
    if preds_out:
        out_path = Path(preds_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as wf:
            for c, r, p in zip(codes, refs, preds):
                wf.write(
                    json.dumps({"code": c, "reference": r, "prediction": p}, ensure_ascii=False)
                    + "\n"
                )
        console.print(Panel.fit(f"📝 Predictions saved to {out_path}", style="bold green"))

    metrics = {"bleu": bleu, "rougeL": rougeL, "bert_score": bert, "codebleu": codebleu}
    console.print(Panel.fit(str(metrics), style="bold green"))

    @main.command()
    @click.option(
        "--index",
        "index_path",
        required=True,
        help="Path to a FAISS index built by build-index (e.g. data/code_retrieval_index.faiss)",
    )
    @click.option("--top-k", type=int, default=3, help="Number of similar examples to retrieve")
    @click.argument("code")
    def query_index(index_path, top_k, code):
        """Query an existing FAISS index with a code snippet and print top matches."""
        from .retrieval import CodeRetriever

        console.print(Panel.fit("🔎 Querying code retrieval index...", style="bold blue"))

        try:
            retriever = CodeRetriever()
            retriever.load_index(index_path)
            matches = retriever.retrieve_similar_code(code, k=top_k)

            table = Table(title=f"Top {top_k} similar code snippets")
            table.add_column("Rank", justify="right", style="cyan")
            table.add_column("Snippet", style="white")
            for i, snippet in enumerate(matches, start=1):
                # Truncate long snippets for display
                preview = snippet if len(snippet) < 280 else snippet[:277] + "..."
                table.add_row(str(i), preview)
            console.print(table)

        except FileNotFoundError as e:
            console.print(Panel.fit(f"❌ {e}", style="bold red"))
        except Exception as e:
            console.print(Panel.fit(f"❌ Failed to query index: {e}", style="bold red"))


@main.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
@click.option(
    "--output-path",
    "-o",
    default="data/code_retrieval_index.faiss",
    help="Path to save the FAISS index",
)
def build_index(config, output_path):
    """Build a FAISS index for code retrieval from the training data."""
    import json

    from .retrieval import CodeRetriever
    from .utils import load_config

    console.print(Panel.fit("🛠️ Building code retrieval index...", style="bold blue"))
    cfg = load_config(config)
    data_path = cfg.get("data", {}).get("train_file")
    if not data_path:
        console.print(Panel.fit("❌ No training data file specified in config.", style="bold red"))
        return

    try:
        with open(data_path, "r") as f:
            data = json.load(f)

        codes = [item["code"] for item in data if "code" in item]

        retriever = CodeRetriever()
        retriever.build_index(codes, save_path=output_path)

        console.print(
            Panel.fit(
                f"✅ Index built successfully with {len(codes)} code snippets and saved to {output_path}",
                style="bold green",
            )
        )
    except Exception as e:
        console.print(Panel.fit(f"❌ Failed to build index: {e}", style="bold red"))


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed validation information")
def validate_config(config_path, verbose):
    """Validate a configuration file."""
    from .config_validator import ConfigValidator

    console.print(Panel.fit("🔍 Validating configuration...", style="bold blue"))

    validator = ConfigValidator()
    is_valid = validator.validate_config(config_path)

    if verbose or not is_valid:
        report = validator.get_validation_report()
        console.print(
            Panel(
                report,
                title="Validation Report",
                border_style=(
                    "yellow" if validator.warnings else ("red" if validator.errors else "green")
                ),
            )
        )

    if is_valid:
        console.print(Panel.fit("✅ Configuration is valid!", style="bold green"))
    else:
        console.print(Panel.fit("❌ Configuration validation failed!", style="bold red"))
        raise click.ClickException("Invalid configuration")


@main.command()
@click.option("--model-path", "-m", default="./results", help="Path to trained model")
@click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
@click.option("--iterations", type=int, default=3, help="Number of iterations per test")
@click.option("--output", "-o", help="Output file for benchmark results")
@click.option("--include-rag", is_flag=True, help="Include Enhanced RAG in benchmarks")
def benchmark(model_path, config, iterations, output, include_rag):
    """Run performance benchmarks on the code explainer."""
    import json

    from .profiler import benchmark_code_explainer

    console.print(Panel.fit("🏃 Starting performance benchmark...", style="bold blue"))

    try:
        with console.status("[bold green]Running benchmarks..."):
            results = benchmark_code_explainer(
                model_path=model_path, config_path=config, num_iterations=iterations
            )

        # Display summary
        summary = results["overall_summary"]
        table = Table(title="Benchmark Summary")
        table.add_column("Operation", justify="left", style="cyan")
        table.add_column("Count", justify="right", style="white")
        table.add_column("Avg Duration (ms)", justify="right", style="green")
        table.add_column("Avg Memory (MB)", justify="right", style="yellow")

        for operation, stats in summary.items():
            if operation != "message":
                table.add_row(
                    operation.replace("_", " ").title(),
                    str(stats["count"]),
                    f"{stats['duration_ms']['mean']:.2f}",
                    f"{stats['memory_mb']['mean']:.1f}",
                )

        console.print(table)

        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(Panel.fit(f"📊 Benchmark results saved to {output}", style="bold green"))

        console.print(Panel.fit("✅ Benchmark completed successfully!", style="bold green"))

    except Exception as e:
        console.print(Panel.fit(f"❌ Benchmark failed: {e}", style="bold red"))
        raise


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file for quality analysis results")
@click.option("--format", type=click.Choice(["json", "text"]), default="text", help="Output format")
@click.option("--show-suggestions", is_flag=True, help="Show improvement suggestions")
def analyze_quality(file_path, output, format, show_suggestions):
    """Analyze code quality and provide improvement suggestions."""
    import json

    from .quality_analyzer import CodeQualityAnalyzer

    console.print(Panel.fit(f"🔍 Analyzing code quality for {file_path}...", style="bold blue"))

    try:
        with open(file_path, "r") as f:
            code = f.read()

        analyzer = CodeQualityAnalyzer()
        analyzer.analyze_code(code)
        results = analyzer.get_summary()

        # Display summary
        table = Table(title="Quality Analysis Summary")
        table.add_column("Level", justify="left", style="cyan")
        table.add_column("Count", justify="right", style="white")

        for level, count in results["by_level"].items():
            if count > 0:
                color = {"critical": "red", "error": "red", "warning": "yellow", "info": "blue"}
                table.add_row(level.upper(), str(count), style=color.get(level, "white"))

        console.print(table)

        # Show issues if any
        if results["total_issues"] > 0:
            issues_table = Table(title="Issues Found")
            issues_table.add_column("Line", justify="right", style="dim")
            issues_table.add_column("Level", justify="center")
            issues_table.add_column("Message", justify="left")
            if show_suggestions:
                issues_table.add_column("Suggestion", justify="left", style="green")

            for issue in results["issues"]:
                line_num = str(issue["line_number"]) if issue["line_number"] else "N/A"
                level_color = {
                    "critical": "red",
                    "error": "red",
                    "warning": "yellow",
                    "info": "blue",
                }
                level_style = level_color.get(issue["level"], "white")

                row = [line_num, issue["level"].upper(), issue["message"]]
                if show_suggestions and issue["suggestion"]:
                    row.append(issue["suggestion"])
                elif show_suggestions:
                    row.append("")

                issues_table.add_row(
                    *row, style=level_style if issue["level"] == "critical" else None
                )

            console.print(issues_table)
        else:
            console.print(
                Panel.fit("🎉 No issues found! Your code looks great!", style="bold green")
            )

        # Save results if requested
        if output:
            if format == "json":
                with open(output, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                # Text format
                with open(output, "w") as f:
                    f.write(f"Code Quality Analysis for {file_path}\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Total issues: {results['total_issues']}\n\n")

                    if results["total_issues"] > 0:
                        for issue in results["issues"]:
                            line_info = (
                                f" (line {issue['line_number']})" if issue["line_number"] else ""
                            )
                            f.write(f"[{issue['level'].upper()}]{line_info} {issue['message']}\n")
                            if issue["suggestion"]:
                                f.write(f"  Suggestion: {issue['suggestion']}\n")
                            f.write("\n")

            console.print(Panel.fit(f"📄 Results saved to {output}", style="bold green"))

    except Exception as e:
        console.print(Panel.fit(f"❌ Quality analysis failed: {e}", style="bold red"))
        raise


@main.command()
@click.option("--cache-dir", default=".cache", help="Cache directory path")
@click.option("--explanations", is_flag=True, help="Clear explanation cache")
@click.option("--embeddings", is_flag=True, help="Clear embedding cache")
@click.option("--all", "clear_all", is_flag=True, help="Clear all caches")
def clear_cache(cache_dir, explanations, embeddings, clear_all):
    """Clear various caches to free up disk space."""
    console.print(Panel.fit("🗑️ Clearing Caches", style="bold yellow"))

    try:
        if clear_all or explanations:
            from .cache import ExplanationCache

            explanation_cache = ExplanationCache(f"{cache_dir}/explanations")
            old_size = explanation_cache.size()
            explanation_cache.clear()
            console.print(f"✅ Cleared {old_size} explanation cache entries")

        if clear_all or embeddings:
            from .cache import EmbeddingCache

            embedding_cache = EmbeddingCache(f"{cache_dir}/embeddings")
            embedding_cache.clear()
            console.print("✅ Cleared embedding cache")

        if not any([explanations, embeddings, clear_all]):
            console.print(
                "[yellow]No cache type specified. Use --explanations, --embeddings, or --all[/yellow]"
            )

    except Exception as e:
        console.print(Panel.fit(f"❌ Cache clearing failed: {e}", style="bold red"))
        raise


@main.command()
@click.option("--cache-dir", default=".cache", help="Cache directory path")
def cache_stats(cache_dir):
    """Show cache statistics and usage information."""
    console.print(Panel.fit("📊 Cache Statistics", style="bold blue"))

    try:
        from .cache import ExplanationCache, EmbeddingCache

        # Explanation cache stats
        explanation_cache = ExplanationCache(f"{cache_dir}/explanations")
        exp_stats = explanation_cache.stats()

        # Embedding cache stats
        embedding_cache = EmbeddingCache(f"{cache_dir}/embeddings")
        embedding_dir = Path(f"{cache_dir}/embeddings")
        embedding_count = len(list(embedding_dir.glob("*.pkl"))) if embedding_dir.exists() else 0

        # Create table
        table = Table(title="Cache Usage")
        table.add_column("Cache Type", style="cyan")
        table.add_column("Entries", style="green")
        table.add_column("Details", style="yellow")

        table.add_row(
            "Explanations",
            str(exp_stats["size"]),
            f"Total accesses: {exp_stats['total_access_count']}",
        )
        table.add_row("Embeddings", str(embedding_count), "Pre-computed code embeddings")

        console.print(table)

        if exp_stats["size"] > 0:
            console.print(f"\n[bold]Explanation Cache Details:[/bold]")
            console.print(f"• Average access count: {exp_stats['avg_access_count']:.1f}")
            console.print(f"• Strategies used: {', '.join(exp_stats['strategies'])}")
            console.print(f"• Models used: {', '.join(exp_stats['models'])}")

    except Exception as e:
        console.print(Panel.fit(f"❌ Failed to get cache stats: {e}", style="bold red"))
        raise


@main.command()
@click.argument("code")
@click.option("--timeout", default=10, help="Execution timeout in seconds")
@click.option("--max-memory", default=100, help="Maximum memory usage in MB")
def safe_execute(code, timeout, max_memory):
    """Safely execute code with security validation and resource limits."""
    console.print(Panel.fit("🔒 Safe Code Execution", style="bold blue"))

    try:
        from .security import SafeCodeExecutor

        executor = SafeCodeExecutor(timeout=timeout, max_memory_mb=max_memory)
        result = executor.execute_code(code)

        if result["success"]:
            console.print("[bold green]✅ Execution successful[/bold green]")
            if result.get("output"):
                console.print(Panel(result["output"], title="Output", border_style="green"))
        else:
            console.print("[bold red]❌ Execution failed[/bold red]")
            if result.get("error"):
                console.print(Panel(result["error"], title="Error", border_style="red"))
            if result.get("stderr"):
                console.print(Panel(result["stderr"], title="Stderr", border_style="red"))

    except Exception as e:
        console.print(Panel.fit(f"❌ Safe execution failed: {e}", style="bold red"))
        raise


@main.command()
@click.argument("code")
def validate_security(code):
    """Validate code for security risks without executing it."""
    console.print(Panel.fit("🛡️ Security Validation", style="bold blue"))

    try:
        from .security import CodeSecurityValidator

        validator = CodeSecurityValidator()
        is_safe, issues = validator.validate_code(code)

        if is_safe:
            console.print("[bold green]✅ Code passed security validation[/bold green]")
        else:
            console.print("[bold red]⚠️ Security issues found:[/bold red]")
            for issue in issues:
                console.print(f"• {issue}")

    except Exception as e:
        console.print(Panel.fit(f"❌ Security validation failed: {e}", style="bold red"))
        raise


if __name__ == "__main__":
    main()
