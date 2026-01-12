"""AST-based code analysis utilities.

Optimized for performance with:
- Single-pass AST traversal combining all extractors
- Tuple-based node type checking (faster than multiple isinstance)
- Pre-cached AST trees to avoid redundant parsing
- Efficient string building with join patterns
"""

import ast
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union, cast

# Pre-define node type tuples for faster isinstance checks
_FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMPORT_TYPES = (ast.Import, ast.ImportFrom)


@dataclass(frozen=True, slots=True)
class ASTInfo:
    """Immutable container for AST analysis results."""
    functions: Tuple[str, ...]
    classes: Tuple[str, ...]
    imports: Tuple[str, ...]
    docstrings: Tuple[str, ...]
    has_recursion: bool = False
    complexity_hints: Tuple[str, ...] = ()


@lru_cache(maxsize=1024)
def _parse_ast_cached(code: str) -> Optional[ast.Module]:
    """Parse code to AST with caching to avoid redundant parsing."""
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return None


@lru_cache(maxsize=512)
def analyze_code_comprehensive(code: str) -> ASTInfo:
    """Single-pass comprehensive AST analysis.
    
    Combines function extraction, class extraction, import analysis,
    and docstring collection into one AST traversal for efficiency.
    """
    tree = _parse_ast_cached(code)
    if tree is None:
        return ASTInfo((), (), (), ())
    
    funcs: List[str] = []
    classes: List[str] = []
    imports_set: set = set()
    docs: List[str] = []
    func_names_set: set = set()  # For recursion detection
    has_recursion = False
    complexity_hints: List[str] = []
    
    # Get module docstring
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        docs.append(f"Module doc: {mod_doc.strip()[:200]}")
    
    # Single-pass traversal extracting all information
    for node in ast.walk(tree):
        node_type = type(node)
        
        # Function handling (combined sync + async)
        if node_type in (ast.FunctionDef, ast.AsyncFunctionDef):
            fn = cast(Union[ast.FunctionDef, ast.AsyncFunctionDef], node)
            funcs.append(fn.name)
            func_names_set.add(fn.name)
            
            # Extract docstring
            doc = ast.get_docstring(fn)
            if doc:
                docs.append(f"Function {fn.name} doc: {doc.strip()[:150]}")
            
            # Detect recursion (function calls itself)
            if not has_recursion:
                for child in ast.walk(fn):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == fn.name:
                            has_recursion = True
                            complexity_hints.append(f"recursive:{fn.name}")
                            break
        
        # Class handling
        elif node_type is ast.ClassDef:
            cls = cast(ast.ClassDef, node)
            classes.append(cls.name)
            doc = ast.get_docstring(cls)
            if doc:
                docs.append(f"Class {cls.name} doc: {doc.strip()[:150]}")
        
        # Import handling (combined Import + ImportFrom)
        elif node_type is ast.Import:
            imports_set.update(alias.name for alias in node.names)
        elif node_type is ast.ImportFrom:
            mod = node.module or ""
            imports_set.update(f"{mod}.{alias.name}" for alias in node.names)
        
        # Complexity hints from loops
        elif node_type in (ast.For, ast.While):
            # Check for nested loops
            for child in ast.walk(node):
                if child is not node and type(child) in (ast.For, ast.While):
                    complexity_hints.append("nested_loop")
                    break
    
    return ASTInfo(
        functions=tuple(funcs),
        classes=tuple(classes),
        imports=tuple(sorted(imports_set)),
        docstrings=tuple(docs),
        has_recursion=has_recursion,
        complexity_hints=tuple(set(complexity_hints))
    )


def _summarize_python_ast(code: str) -> str:
    """Summarize Python code structure via AST for prompting.
    Returns a concise, bullet-style summary.
    Uses cached comprehensive analysis for efficiency.
    """
    info = analyze_code_comprehensive(code)
    if not info.functions and not info.classes and not info.imports:
        return ""
    
    lines = [
        "Context:",
        f"- Functions: {', '.join(info.functions) if info.functions else 'none'}",
        f"- Classes: {', '.join(info.classes) if info.classes else 'none'}",
        f"- Imports: {', '.join(info.imports) if info.imports else 'none'}",
    ]
    
    if info.has_recursion:
        lines.append("- Note: Contains recursive function(s)")
    
    return "\n".join(lines)


@lru_cache(maxsize=512)
def _extract_python_ast_info(code: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (functions, classes, imports) lists for Python code via AST.
    Delegates to comprehensive analysis for cache efficiency.
    """
    info = analyze_code_comprehensive(code)
    return list(info.functions), list(info.classes), list(info.imports)


@lru_cache(maxsize=256)
def _collect_docstrings_from_code(code: str) -> List[str]:
    """Collect module, class, and function docstrings from the snippet.
    Delegates to comprehensive analysis for cache efficiency.
    """
    info = analyze_code_comprehensive(code)
    return list(info.docstrings)


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
        except (ImportError, AttributeError, ModuleNotFoundError):
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