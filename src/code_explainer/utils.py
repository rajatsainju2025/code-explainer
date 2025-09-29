"""Configuration management utilities."""

import json
import logging
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from functools import lru_cache

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=getattr(logging, level.upper()), format=log_format, handlers=handlers)


def get_device() -> str:
    """Get the best available device for training/inference.

    This function is maintained for backwards compatibility.
    For new code, consider using DeviceManager directly.
    """
    try:
        from .device_manager import device_manager
        device_capabilities = device_manager.get_optimal_device()
        return device_capabilities.device_type
    except Exception:
        # Fallback to original logic if DeviceManager fails
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


def detect_language(code: str) -> str:
    """Very simple language detector for code snippets.
    Returns one of: python, javascript, java, cpp.
    """
    code_l = code.lower()
    if "#include" in code_l or "std::" in code or ";" in code and "using namespace" in code_l:
        return "cpp"
    if "public static void main" in code_l or "class " in code and "system.out" in code_l:
        return "java"
    if "function " in code_l or "=>" in code or "console.log" in code_l:
        return "javascript"
    return "python"


def _summarize_python_ast(code: str) -> str:
    """Summarize Python code structure via AST for prompting.
    Returns a concise, bullet-style summary.
    """
    import ast

    try:
        tree = ast.parse(code)
    except Exception:
        return ""

    funcs: List[str] = []
    classes: List[str] = []
    imports: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                else:
                    mod = node.module or ""
                    imports.extend([f"{mod}.{alias.name}" for alias in node.names])
            except Exception:
                pass

    lines = [
        "Context:",
        f"- Functions: {', '.join(funcs) if funcs else 'none'}",
        f"- Classes: {', '.join(classes) if classes else 'none'}",
        f"- Imports: {', '.join(sorted(set(imports))) if imports else 'none'}",
    ]
    return "\n".join(lines)


@lru_cache(maxsize=512)
def _extract_python_ast_info(code: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (functions, classes, imports) lists for Python code via AST."""
    try:
        tree = ast.parse(code)
    except Exception:
        return [], [], []

    funcs: List[str] = []
    classes: List[str] = []
    imports: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                else:
                    mod = node.module or ""
                    imports.extend([f"{mod}.{alias.name}" for alias in node.names])
            except Exception:
                pass
    return funcs, classes, imports


@lru_cache(maxsize=256)
def _collect_docstrings_from_code(code: str) -> List[str]:
    """Collect module, class, and function docstrings from the snippet."""
    docs: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return docs

    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        docs.append(f"Module doc: {mod_doc.strip()}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, getattr(ast, "AsyncFunctionDef", type("X", (), {})))):
            fn = cast(Union[ast.FunctionDef, Any], node)
            doc = ast.get_docstring(fn)
            if doc:
                name = getattr(fn, "name", "<fn>")
                docs.append(f"Function {name} doc: {doc.strip()}")
        elif isinstance(node, ast.ClassDef):
            cls = cast(ast.ClassDef, node)
            doc = ast.get_docstring(cls)
            if doc:
                docs.append(f"Class {cls.name} doc: {doc.strip()}")
    return docs


def _collect_import_docs(
    imports: List[str], max_modules: int = 5, max_chars: int = 600
) -> List[str]:
    """Attempt to import a small allowlist of stdlib modules and extract short docs."""
    import importlib

    allow = {
        "math",
        "re",
        "itertools",
        "functools",
        "collections",
        "statistics",
        "random",
        "json",
        "string",
        "datetime",
        "heapq",
        "bisect",
    }
    docs: List[str] = []
    count = 0
    for name in imports:
        root = name.split(".")[0]
        if root not in allow:
            continue
        try:
            mod = importlib.import_module(root)
            doc = (getattr(mod, "__doc__", None) or "").strip()
            if doc:
                snippet = doc.split("\n")[:6]
                text = " ".join(line.strip() for line in snippet)
                if text:
                    docs.append(f"Doc({root}): {text[:max_chars]}")
                    count += 1
                    if count >= max_modules:
                        break
        except Exception:
            continue
    return docs


def summarize_code_structure(code: str) -> str:
    """Return a concise textual summary of code structure when possible.
    Currently supports Python via AST; returns empty string for other languages.
    """
    lang = detect_language(code)
    if lang == "python":
        return _summarize_python_ast(code)
    return ""


def _safe_exec_subprocess(code: str, timeout_s: float = 1.0, mem_mb: int = 64) -> Tuple[str, str]:
    """Run code in a subprocess with basic time/memory limits and capture output.
    Returns (stdout, stderr) truncated.
    """
    import os
    import shlex
    import subprocess
    import sys
    import tempfile
    import textwrap

    # Wrapper to set resource limits (Unix only)
    prelude = (
        "import sys,resource,os\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({int(timeout_s)}, {int(timeout_s)}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({mem_mb*1024*1024}, {mem_mb*1024*1024}))\n"
        "os.environ.clear()\n"
        "\n"
    )

    wrapped = prelude + code
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            input=None,
            capture_output=True,
            text=True,
            timeout=max(timeout_s, 0.1),
            cwd=tempfile.gettempdir(),
            env={},
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
    except subprocess.TimeoutExpired:
        out, err = "", "TimeoutExpired"
    except Exception as e:
        out, err = "", f"ExecutionError: {e}"

    # Truncate
    def _trunc(s: str, n: int = 500) -> str:
        return (s[:n] + "…") if len(s) > n else s

    return _trunc(out), _trunc(err)


def prompt_for_language(config: Dict[str, Any], code: str) -> str:
    lang = detect_language(code)
    # Support both legacy 'prompt' and new 'prompting' sections
    prompt_cfg = config.get("prompt") or config.get("prompting") or {}
    templates = prompt_cfg.get("language_templates", {})
    default_template = prompt_cfg.get("template", f"Explain the following {lang} code:\n{{code}}")
    base_template = templates.get(lang, default_template)
    base_prompt = base_template.format(code=code.strip())

    strategy = prompt_cfg.get("strategy", "vanilla")

    # AST-augmented
    if strategy == "ast_augmented" and lang == "python":
        ctx = summarize_code_structure(code)
        if ctx:
            return (
                "You are a helpful assistant that explains code clearly and concisely.\n"
                + ctx
                + "\n\n"
                + base_prompt
            )

    # Retrieval-augmented: use docstrings (own code + limited stdlib import docs)
    if strategy == "retrieval_augmented" and lang == "python":
        funcs, classes, imports = _extract_python_ast_info(code)
        own_docs = _collect_docstrings_from_code(code)
        import_docs = _collect_import_docs(imports)
        retrieved: List[str] = []
        if own_docs:
            retrieved.append("Own docstrings:\n- " + "\n- ".join(own_docs[:6]))
        if import_docs:
            retrieved.append("Imports docs:\n- " + "\n- ".join(import_docs))
        if retrieved:
            return (
                "You are a helpful assistant that uses the following retrieved context to explain the code.\n"
                + "\n\n".join(retrieved)
                + "\n\n"
                + base_prompt
            )

    # Execution-trace augmented: run safely and include observed stdout/stderr
    if strategy == "execution_trace" and lang == "python":
        out, err = _safe_exec_subprocess(code)
        trace_lines = ["Execution trace (safe sandbox):"]
        if out:
            trace_lines.append("stdout: " + out)
        if err:
            trace_lines.append("stderr: " + err)
        trace = "\n".join(trace_lines)
        return (
            "You are a helpful assistant. Use the execution trace to explain behavior, but do not assume more than observed.\n"
            + trace
            + "\n\n"
            + base_prompt
        )

    # Enhanced RAG with code retrieval
    if strategy == "enhanced_rag" and lang == "python":
        from .retrieval import CodeRetriever

        retriever = CodeRetriever()
        try:
            retriever.load_index(
                config.get("retrieval", {}).get("index_path", "data/code_retrieval_index.faiss")
            )
            similar_codes = retriever.retrieve_similar_code(code, k=3)

            rag_context = [
                "You are a helpful assistant that uses the following similar code examples to provide a better explanation.",
                "---",
                "Similar Code Examples:",
            ]
            for i, example in enumerate(similar_codes):
                rag_context.append(f"\nExample {i+1}:\n```python\n{example}\n```")

            rag_context.append("---\n")
            rag_context.append(base_prompt)

            return "\n".join(rag_context)

        except Exception as e:
            logging.warning(f"Enhanced RAG failed: {e}. Falling back to vanilla prompt.")
            # Fallback to vanilla if retrieval fails
            return base_prompt

    return base_prompt
