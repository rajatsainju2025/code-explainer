"""Prompt generation utilities with optimized caching and string handling."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List

from .ast_analysis import _collect_docstrings_from_code, _collect_import_docs, _extract_python_ast_info, summarize_code_structure
from .execution import _safe_exec_subprocess
from .language import detect_language

if TYPE_CHECKING:
    from ..retrieval.retriever import CodeRetriever

# Pre-compute strategy constants for O(1) comparison via interning
_STRATEGY_AST: str = "ast_augmented"
_STRATEGY_RAG: str = "retrieval_augmented"
_STRATEGY_EXEC: str = "execution_trace"
_STRATEGY_ENHANCED: str = "enhanced_rag"
_STRATEGY_VANILLA: str = "vanilla"

# Pre-computed prompt prefixes (avoid repeated string allocation)
_AST_PREFIX: str = "You are a helpful assistant that explains code clearly and concisely.\n"
_RAG_PREFIX: str = "You are a helpful assistant that uses the following retrieved context to explain the code.\n"
_EXEC_PREFIX: str = "You are a helpful assistant. Use the execution trace to explain behavior, but do not assume more than observed.\n"
_ENHANCED_PREFIX: str = "You are a helpful assistant that uses the following similar code examples to provide a better explanation."

# Retriever singleton for enhanced RAG (lazy loaded)
_retriever_instance: "CodeRetriever | None" = None


@lru_cache(maxsize=64)
def _cached_default_template(lang: str) -> str:
    """Cache default templates per language to avoid f-string allocation."""
    return f"Explain the following {lang} code:\n{{code}}"


def _get_retriever() -> "CodeRetriever":
    """Get or create singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        from ..retrieval.retriever import CodeRetriever
        _retriever_instance = CodeRetriever()
    return _retriever_instance


def _build_rag_context(own_docs: List[str], import_docs: List[str]) -> str:
    """Build RAG context string efficiently using list join."""
    parts: List[str] = []
    if own_docs:
        # Limit to 6 own docs, pre-slice
        limited_docs = own_docs[:6] if len(own_docs) > 6 else own_docs
        parts.append("Own docstrings:\n- " + "\n- ".join(limited_docs))
    if import_docs:
        # Limit import docs to 8 items for performance
        limited_imports = import_docs[:8] if len(import_docs) > 8 else import_docs
        parts.append("Imports docs:\n- " + "\n- ".join(limited_imports))
    return "\n".join(parts) if parts else ""


def _build_exec_trace(out: str, err: str) -> str:
    """Build execution trace string efficiently."""
    parts = ["Execution trace (safe sandbox):"]
    if out:
        parts.append(f"stdout: {out}")
    if err:
        parts.append(f"stderr: {err}")
    return "\n".join(parts)


def prompt_for_language(config: Dict[str, Any], code: str) -> str:
    """Generate a prompt for code explanation based on configuration and code content.
    
    Optimized with:
    - Cached default templates per language
    - Pre-computed prefix strings
    - Early returns to minimize work
    - Efficient string building via list joins
    - Singleton retriever for enhanced RAG
    """
    lang = detect_language(code)

    # Support both legacy 'prompt' and new 'prompting' sections
    prompt_cfg = config.get("prompt") or config.get("prompting") or {}
    strategy = prompt_cfg.get("strategy", _STRATEGY_VANILLA)
    
    # Early exit for vanilla strategy (most common path)
    templates = prompt_cfg.get("language_templates", {})
    default_template = prompt_cfg.get("template") or _cached_default_template(lang)
    base_template = templates.get(lang, default_template)
    code_stripped = code.strip()
    base_prompt = base_template.format(code=code_stripped)

    # Skip augmentation for non-Python code or vanilla strategy
    if lang != "python" or strategy == _STRATEGY_VANILLA:
        return base_prompt

    # Dispatch based on strategy using dict lookup for O(1)
    if strategy == _STRATEGY_AST:
        ctx = summarize_code_structure(code)
        if ctx:
            return f"{_AST_PREFIX}{ctx}\n\n{base_prompt}"
        return base_prompt

    if strategy == _STRATEGY_RAG:
        _funcs, _classes, imports = _extract_python_ast_info(code)
        own_docs = _collect_docstrings_from_code(code)
        import_docs = _collect_import_docs(imports)
        rag_context = _build_rag_context(own_docs, import_docs)
        if rag_context:
            return f"{_RAG_PREFIX}{rag_context}\n\n{base_prompt}"
        return base_prompt

    if strategy == _STRATEGY_EXEC:
        out, err = _safe_exec_subprocess(code)
        trace = _build_exec_trace(out, err)
        return f"{_EXEC_PREFIX}{trace}\n\n{base_prompt}"

    if strategy == _STRATEGY_ENHANCED:
        try:
            retriever = _get_retriever()
            index_path = config.get("retrieval", {}).get("index_path", "data/code_retrieval_index.faiss")
            retriever.load_index(index_path)
            similar_codes = retriever.retrieve_similar_code(code, k=3)

            # Build efficiently using list
            rag_parts = [
                _ENHANCED_PREFIX,
                "---",
                "Similar Code Examples:",
            ]
            for i, example in enumerate(similar_codes, 1):
                rag_parts.append(f"\nExample {i}:\n```python\n{example}\n```")

            rag_parts.append("---\n")
            rag_parts.append(base_prompt)
            return "\n".join(rag_parts)

        except (ImportError, FileNotFoundError, RuntimeError) as e:
            logging.warning("Enhanced RAG failed: %s. Falling back to vanilla prompt.", e)

    return base_prompt