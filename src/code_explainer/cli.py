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
@click.option("--self-consistency", type=int, default=0, show_default=True, help="If >0, generate N samples per item and compute self-consistency metrics")
def eval(model_path, config, test_file, preds_out, max_samples, prompt_strategy, self_consistency):
    """Evaluate a model on a test set (BLEU/ROUGE/BERTScore) and optionally save predictions.

    Supports input as a JSON array file or JSONL (one JSON object per line).
    Each item should include at least {"code": str, "explanation": str}.
    """
    import json

    from .metrics.evaluate import (
        compute_bleu,
        compute_codebert_score,
        compute_codebleu,
        compute_rouge_l,
    )
    from .metrics.provenance import provenance_scores
    from .model import CodeExplainer
    from .metrics.self_consistency import pairwise_scores as sc_pairwise

    console.print(Panel.fit("📏 Running evaluation", style="bold blue"))
    explainer = CodeExplainer(model_path=model_path, config_path=config)

    # Load test data (JSON array or JSONL)
    if test_file is None:
        cfg = explainer.config
        test_file = cfg.get("data", {}).get("test_file")
    if not test_file:
        console.print(Panel.fit("❌ No test file provided or configured.", style="bold red"))
        return

    def _read_json_or_jsonl(path: str):
        p = Path(path)
        if p.suffix.lower() == ".jsonl":
            items = []
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        # skip malformed lines
                        continue
            return items
        else:
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)

    data = _read_json_or_jsonl(test_file)

    if max_samples is not None:
        try:
            k = int(max_samples)
            data = data[: max(0, k)]
        except Exception:
            pass

    refs = []
    preds = []
    codes = []
    prov_precisions = []
    prov_recalls = []
    for ex in data:
        code = ex.get("code", "")
        ref = ex.get("explanation", "")
        # Optional: per-example allowed source IDs for provenance
        source_ids = ex.get("source_ids") or ex.get("sources") or []
        try:
            if self_consistency and int(self_consistency) > 0:
                # Generate multiple samples and reduce to a canonical prediction (e.g., first), but keep SC metrics
                outs = []
                for _ in range(int(self_consistency)):
                    try:
                        outs.append(explainer.explain_code(code, strategy=prompt_strategy))
                    except Exception:
                        outs.append("")
                sc = sc_pairwise(outs)
                # Attach SC scores into per-example lists for later averaging
                # store as pseudo provenance arrays
                ex["_sc_bleu"] = sc.avg_pairwise_bleu
                ex["_sc_rougeL"] = sc.avg_pairwise_rougeL
                pred = outs[0] if outs else ""
            else:
                pred = explainer.explain_code(code, strategy=prompt_strategy)
        except Exception:
            pred = ""
        codes.append(code)
        refs.append(ref)
        preds.append(pred)
        # provenance per example if we have source ids
        if isinstance(source_ids, (list, tuple)) and source_ids:
            ps = provenance_scores(pred, [str(s) for s in source_ids])
            prov_precisions.append(ps.precision)
            prov_recalls.append(ps.recall)

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
    if prov_precisions:
        table.add_row("Prov. precision", f"{sum(prov_precisions)/len(prov_precisions):.4f}")
    if prov_recalls:
        table.add_row("Prov. recall", f"{sum(prov_recalls)/len(prov_recalls):.4f}")
    # Show SC metrics if computed
    sc_bleus = [ex.get("_sc_bleu") for ex in data if isinstance(ex, dict) and ex.get("_sc_bleu") is not None]
    sc_rouges = [ex.get("_sc_rougeL") for ex in data if isinstance(ex, dict) and ex.get("_sc_rougeL") is not None]
    sc_bleus_f = [float(x) for x in sc_bleus if isinstance(x, (int, float))]
    sc_rouges_f = [float(x) for x in sc_rouges if isinstance(x, (int, float))]
    if sc_bleus_f:
        table.add_row("Self-consistency BLEU", f"{sum(sc_bleus_f)/len(sc_bleus_f):.4f}")
    if sc_rouges_f:
        table.add_row("Self-consistency ROUGE-L", f"{sum(sc_rouges_f)/len(sc_rouges_f):.4f}")
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
    if prov_precisions:
        metrics["provenance_precision"] = sum(prov_precisions) / len(prov_precisions)
    if prov_recalls:
        metrics["provenance_recall"] = sum(prov_recalls) / len(prov_recalls)
    # Aggregate self-consistency metrics if present
    if sc_bleus_f:
        metrics["self_consistency_bleu"] = sum(sc_bleus_f) / len(sc_bleus_f)
    if sc_rouges_f:
        metrics["self_consistency_rougeL"] = sum(sc_rouges_f) / len(sc_rouges_f)
    console.print(Panel.fit(str(metrics), style="bold green"))

    @main.command()
    @click.option(
        "--index",
        "index_path",
        required=True,
        help="Path to a FAISS index built by build-index (e.g. data/code_retrieval_index.faiss)",
    )
    @click.option("--top-k", type=int, default=3, help="Number of similar examples to retrieve")
    @click.option(
        "--method",
        type=click.Choice(["faiss", "bm25", "hybrid"]),
        default="hybrid",
        show_default=True,
        help="Retrieval method",
    )
    @click.option(
        "--alpha",
        type=float,
        default=0.5,
        show_default=True,
        help="Hybrid weight toward FAISS similarity (0..1)",
    )
    @click.argument("code")
    def query_index(index_path, top_k, method, alpha, code):
        """Query an existing FAISS index with a code snippet and print top matches."""
        from .retrieval import CodeRetriever

        console.print(Panel.fit("🔎 Querying code retrieval index...", style="bold blue"))

        try:
            retriever = CodeRetriever()
            retriever.load_index(index_path)
            matches = retriever.retrieve_similar_code(code, k=top_k, method=method, alpha=alpha)

            table = Table(title=f"Top {top_k} similar code snippets ({method}, alpha={alpha})")
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
@click.option("--eval-jsonl", required=True, help="JSONL file for eval (id, code)")
@click.option("--train-jsonl", required=True, help="JSONL file for train (id, code)")
@click.option("--ngram", default=5, show_default=True, help="N-gram size for overlap")
@click.option("--threshold", default=0.8, show_default=True, help="Jaccard threshold")
def detect_contamination(eval_jsonl, train_jsonl, ngram, threshold):
    """Detect potential train/eval overlap to reduce contamination risk."""
    from .contamination import detect_overlap, read_jsonl

    console.print(Panel.fit("🧪 Detecting potential contamination", style="bold blue"))
    eval_pairs = read_jsonl(Path(eval_jsonl))
    train_pairs = read_jsonl(Path(train_jsonl))

    flagged = detect_overlap(eval_pairs, train_pairs, ngram=ngram, threshold=threshold)
    if not flagged:
        console.print(Panel.fit("✅ No high-overlap pairs detected.", style="bold green"))
        return

    table = Table(title="Flagged overlaps (descending Jaccard)")
    table.add_column("Eval ID", style="cyan")
    table.add_column("Train ID", style="magenta")
    table.add_column("Jaccard", justify="right")
    for item in flagged[:50]:
        (eid, tid) = item.pair
        table.add_row(eid, tid, f"{item.jaccard:.3f}")
    console.print(table)


@main.command()
@click.option("--num-samples", default=5, show_default=True, help="Number of generations")
@click.option("--strategy", default="vanilla")
@click.argument("code")
def self_consistency(num_samples, strategy, code):
    """Generate multiple explanations and compute self-consistency metrics."""
    from .metrics.self_consistency import pairwise_scores

    explainer = CodeExplainer()
    outs = []
    for _ in range(max(1, int(num_samples))):
        try:
            outs.append(explainer.explain_code(code, strategy=strategy))
        except Exception:
            outs.append("")

    sc = pairwise_scores(outs)
    table = Table(title="Self-Consistency")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("avg_pairwise_bleu", f"{sc.avg_pairwise_bleu:.4f}")
    table.add_row("avg_pairwise_rougeL", f"{sc.avg_pairwise_rougeL:.4f}")
    table.add_row("n_samples", str(sc.n_samples))
    console.print(table)


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


@main.command()
@click.option("--dataset", required=True, help="Open-eval dataset id (e.g., demo-addsub)")
@click.option("--model-path", default="./results")
@click.option("--config", default="configs/default.yaml")
@click.option("--out-csv", default=None)
@click.option("--out-json", default=None)
def eval_open(dataset, model_path, config, out_csv, out_json):
    """Run an open-eval dataset and emit metrics and optional outputs."""
    from .open_evals import run_eval as run_open_eval

    console.print(Panel.fit(f"📊 Running open-eval: {dataset}", style="bold blue"))
    metrics = run_open_eval(dataset, model_path=model_path, config_path=config, out_csv=out_csv, out_json=out_json)
    console.print(Panel.fit(str(metrics), style="bold green"))


@main.command()
@click.option("--test-data", "-t", required=True, help="Path to test data (JSON/JSONL)")
@click.option("--predictions", "-p", required=True, help="Path to model predictions (JSON/JSONL)")
@click.option("--output", "-o", default="llm_judge_report.json", help="Output report path")
@click.option("--judges", multiple=True, default=["gpt-4", "claude-3-sonnet"], help="LLM judges to use")
@click.option("--criteria", multiple=True, default=["accuracy", "clarity", "completeness"], help="Evaluation criteria")
@click.option("--require-consensus", is_flag=True, help="Require consensus among judges")
def eval_llm_judge(test_data, predictions, output, judges, criteria, require_consensus):
    """Run LLM-as-a-Judge evaluation."""
    from .evaluation.llm_judge import run_llm_judge_evaluation
    console.print(Panel.fit("🤖 Running LLM-as-a-Judge Evaluation", style="bold blue"))

    try:
        report = run_llm_judge_evaluation(
            test_file=test_data,
            predictions_file=predictions,
            output_file=output,
            judges=list(judges),
            criteria=list(criteria),
            require_consensus=require_consensus,
        )

        console.print("[bold green]✅ LLM judge evaluation completed![/bold green]")
        console.print(f"Overall Score: {report.get('overall_score', 0.0):.3f}")
        console.print(f"Agreement Rate: {report.get('judge_agreement', 0.0):.3f}")
        console.print(f"Total Evaluations: {report.get('total_evaluations', 0)}")
        console.print(f"Report saved to: {output}")
    except Exception as e:
        console.print(Panel.fit(f"❌ LLM judge evaluation failed: {e}", style="bold red"))
        raise


@main.command()
@click.option("--test-data", "-t", required=True, help="Path to test data (JSON/JSONL)")
@click.option("--predictions-a", "-a", required=True, help="Path to first set of predictions")
@click.option("--predictions-b", "-b", required=True, help="Path to second set of predictions")
@click.option("--output", "-o", default="preference_report.json", help="Output report path")
@click.option("--judges", multiple=True, default=["gpt-4"], help="LLM judges to use")
@click.option("--criteria", multiple=True, default=["overall_quality"], help="Comparison criteria")
@click.option("--use-bradley-terry", is_flag=True, help="Use Bradley-Terry ranking model")
@click.option("--criteria-file", type=click.Path(exists=True), default=None, help="YAML file with constitutional/principles-based criteria")
def eval_preference(test_data, predictions_a, predictions_b, output, judges, criteria, use_bradley_terry, criteria_file):
    """Run preference-based evaluation between two models."""
    import os
    from pathlib import Path
    from .evaluation.preference import run_preference_evaluation

    console.print(Panel.fit("⚖️ Running Preference-Based Evaluation", style="bold blue"))

    try:
        # Build judge config from the first judge option
        if not judges:
            raise click.ClickException("Please specify at least one judge model (e.g., --judges gpt-4)")
        judge_model = list(judges)[0]

        if judge_model.startswith("gpt"):
            judge_config = {
                "type": "openai",
                "model": judge_model,
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }
        elif judge_model.startswith("claude"):
            judge_config = {
                "type": "anthropic",
                "model": judge_model,
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            }
        else:
            judge_config = {"type": "openai", "model": judge_model, "api_key": os.environ.get("OPENAI_API_KEY")}

        system_names = [Path(predictions_a).stem, Path(predictions_b).stem]

        summary = run_preference_evaluation(
            predictions_files=[predictions_a, predictions_b],
            system_names=system_names,
            judge_config=judge_config,
            output_file=output,
            criteria_file=criteria_file,
        )

        console.print("[bold green]✅ Preference evaluation completed![/bold green]")
        # Basic summary printout
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                console.print(f"{k}: {v}")
            else:
                console.print(f"{k}: {v}")
        console.print(f"Report saved to: {output}")
    except Exception as e:
        console.print(Panel.fit(f"❌ Preference evaluation failed: {e}", style="bold red"))
        raise


@main.command()
@click.option("--train-data", "-tr", required=True, help="Path to training data (JSON/JSONL)")
@click.option("--test-data", "-te", required=True, help="Path to test data (JSON/JSONL)")
@click.option("--output", "-o", default="contamination_report.json", help="Output report path")
@click.option("--methods", multiple=True, default=["exact", "ngram", "substring"], help="Detection methods")
@click.option("--fields", multiple=True, default=["code", "explanation"], help="Fields to check")
@click.option("--include-semantic", is_flag=True, help="Include semantic similarity detection (requires sentence-transformers)")
def eval_contamination(train_data, test_data, output, methods, fields, include_semantic):
    """Run contamination detection between train and test data."""
    from .evaluation.contamination import run_contamination_detection
    
    console.print(Panel.fit("🔍 Running Contamination Detection", style="bold blue"))
    
    try:
        detection_methods = list(methods)
        if include_semantic:
            detection_methods.append("semantic")
        
        report = run_contamination_detection(
            train_file=train_data,
            test_file=test_data,
            output_file=output,
            methods=detection_methods,
            fields=list(fields)
        )
        
        console.print(f"[bold green]✅ Contamination detection completed![/bold green]")
        console.print(f"Total Test Examples: {report.total_test_examples}")
        console.print(f"Contaminated Examples: {len(report.contaminated_examples)}")
        console.print(f"Contamination Rate: {report.contamination_rate:.3%}")
        
        if report.contaminated_examples:
            console.print("[bold yellow]⚠️ Contamination detected![/bold yellow]")
            
            # Show contamination by method
            contamination_by_method = report.summary_stats.get("contamination_by_method", {})
            if contamination_by_method:
                console.print("Contamination by method:")
                for method, count in contamination_by_method.items():
                    console.print(f"  {method}: {count}")
        else:
            console.print("[bold green]✅ No contamination detected[/bold green]")
        
        console.print(f"Report saved to: {output}")
        
    except Exception as e:
        console.print(Panel.fit(f"❌ Contamination detection failed: {e}", style="bold red"))
        raise


@main.command()
@click.option("--test-data", "-t", required=True, help="Path to test data (JSON/JSONL)")
@click.option("--model-path", "-m", default="./results", help="Path to trained model")
@click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
@click.option("--output", "-o", default="robustness_report.json", help="Output report path")
@click.option("--test-types", multiple=True, default=["typo", "case", "whitespace", "punctuation"], help="Robustness test types")
@click.option("--severity-levels", multiple=True, type=float, default=[0.05, 0.1, 0.2], help="Severity levels for tests")
@click.option("--max-examples", type=int, default=100, help="Maximum examples to test")
@click.option("--random-seed", type=int, default=42, help="Random seed for reproducibility")
def eval_robustness(test_data, model_path, config, output, test_types, severity_levels, max_examples, random_seed):
    """Run robustness testing on model predictions."""
    from .evaluation.robustness import run_robustness_tests
    from .model import CodeExplainer
    import json
    
    console.print(Panel.fit("🛡️ Running Robustness Testing", style="bold blue"))
    
    try:
        # Load test data
        with open(test_data, 'r') as f:
            if test_data.endswith('.jsonl'):
                examples = [json.loads(line) for line in f]
            else:
                examples = json.load(f)
        
        # Create prediction function
        explainer = CodeExplainer(model_path=model_path, config_path=config)
        
        def predict_func(example):
            return explainer.explain_code(example.get('code', ''))
        
        # Run robustness tests
        report = run_robustness_tests(
            examples=examples,
            predict_func=predict_func,
            output_file=output,
            test_types=list(test_types),
            severity_levels=list(severity_levels),
            max_examples=max_examples,
            random_seed=random_seed
        )
        
        console.print(f"[bold green]✅ Robustness testing completed![/bold green]")
        console.print(f"Total Tests: {report.total_tests}")
        console.print(f"Overall Robustness Score: {report.overall_robustness_score:.3f}")
        
        # Show results by test type
        for test_name, summary in report.test_summaries.items():
            console.print(f"{test_name}: {summary['mean_score']:.3f} ± {summary['std_score']:.3f}")
        
        console.print(f"Report saved to: {output}")
        
    except Exception as e:
        console.print(Panel.fit(f"❌ Robustness testing failed: {e}", style="bold red"))
        raise


if __name__ == "__main__":
    main()
