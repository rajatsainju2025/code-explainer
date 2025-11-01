"""Prompt generation utilities."""

import logging
from typing import Any, Dict, List

from .ast_analysis import _collect_docstrings_from_code, _collect_import_docs, _extract_python_ast_info, summarize_code_structure
from .execution import _safe_exec_subprocess
from .language import detect_language


def prompt_for_language(config: Dict[str, Any], code: str) -> str:
    """Generate a prompt for code explanation based on configuration and code content."""
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
        from ..retrieval import CodeRetriever

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

        except (ImportError, FileNotFoundError, RuntimeError) as e:
            logging.warning(f"Enhanced RAG failed: {e}. Falling back to vanilla prompt.")
            # Fallback to vanilla if retrieval fails
            return base_prompt

    return base_prompt