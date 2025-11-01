"""AST-based code analysis utilities."""

import ast
from functools import lru_cache
from typing import Any, List, Tuple, Union, cast


def _summarize_python_ast(code: str) -> str:
    """Summarize Python code structure via AST for prompting.
    Returns a concise, bullet-style summary.
    """
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
    imports_set = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                if isinstance(node, ast.Import):
                    imports_set.update(alias.name for alias in node.names)
                else:
                    mod = node.module or ""
                    imports_set.update(f"{mod}.{alias.name}" for alias in node.names)
            except Exception:
                pass
    return funcs, classes, list(imports_set)


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
    from .language import detect_language

    lang = detect_language(code)
    if lang == "python":
        return _summarize_python_ast(code)
    return ""