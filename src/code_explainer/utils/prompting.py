"""Prompt generation utilities."""

import logging
from typing import Any, Dict, List

from .ast_analysis import _collect_docstrings_from_code, _collect_import_docs, _extract_python_ast_info, summarize_code_structure
from .execution import _safe_exec_subprocess
from .language import detect_language

# Pre-compute strategy constants for O(1) comparison
_STRATEGY_AST = "ast_augmented"
_STRATEGY_RAG = "retrieval_augmented"
_STRATEGY_EXEC = "execution_trace"
_STRATEGY_ENHANCED = "enhanced_rag"


def prompt_for_language(config: Dict[str, Any], code: str) -> str:
    """Generate a prompt for code explanation based on configuration and code content."""
    lang = detect_language(code)

    # Support both legacy 'prompt' and new 'prompting' sections
    prompt_cfg = config.get("prompt") or config.get("prompting") or {}
    templates = prompt_cfg.get("language_templates", {})
    default_template = prompt_cfg.get("template", f"Explain the following {lang} code:\n{{code}}")
    base_template = templates.get(lang, default_template)
    code_stripped = code.strip()
    base_prompt = base_template.format(code=code_stripped)

    strategy = prompt_cfg.get("strategy", "vanilla")

    # Skip augmentation for non-Python code
    if lang != "python":
        return base_prompt

    # AST-augmented
    if strategy == _STRATEGY_AST:
        ctx = summarize_code_structure(code)
        if ctx:
            return f"You are a helpful assistant that explains code clearly and concisely.\n{ctx}\n\n{base_prompt}"

    # Retrieval-augmented: use docstrings (own code + limited stdlib import docs)
    if strategy == _STRATEGY_RAG:
        funcs, classes, imports = _extract_python_ast_info(code)
        own_docs = _collect_docstrings_from_code(code)
        import_docs = _collect_import_docs(imports)
        retrieved: List[str] = []
        if own_docs:
            retrieved.append("Own docstrings:\n- " + "\n- ".join(own_docs[:6]))
        if import_docs:
            retrieved.append("Imports docs:\n- " + "\n- ".join(import_docs))
        if retrieved:
            return f"You are a helpful assistant that uses the following retrieved context to explain the code.\n{chr(10).join(retrieved)}\n\n{base_prompt}"

    # Execution-trace augmented: run safely and include observed stdout/stderr
    if strategy == _STRATEGY_EXEC:
        out, err = _safe_exec_subprocess(code)
        trace_parts = ["Execution trace (safe sandbox):"]
        if out:
            trace_parts.append(f"stdout: {out}")
        if err:
            trace_parts.append(f"stderr: {err}")
        trace = "\n".join(trace_parts)
        return f"You are a helpful assistant. Use the execution trace to explain behavior, but do not assume more than observed.\n{trace}\n\n{base_prompt}"

    # Enhanced RAG with code retrieval
    if strategy == _STRATEGY_ENHANCED:
        from ..retrieval.retriever import CodeRetriever

        retriever = CodeRetriever()
        try:
            retriever.load_index(
                config.get("retrieval", {}).get("index_path", "data/code_retrieval_index.faiss")
            )
            similar_codes = retriever.retrieve_similar_code(code, k=3)

            rag_parts = [
                "You are a helpful assistant that uses the following similar code examples to provide a better explanation.",
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

    return base_prompt

    return base_prompt