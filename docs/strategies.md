# Strategies

This project supports multiple prompt strategies to improve code explanations. You can set the default in `configs/*.yaml` under `prompt.strategy`, override via CLI (`--prompt-strategy`), or API (`{"code": "...", "strategy": "..."}`).

Supported strategies:

- vanilla
  - Uses the base language-specific template only.
- ast_augmented (Python)
  - Adds a concise AST-derived context (functions, classes, imports) to the prompt.
  - Benefits: better structure awareness and grounding.
- retrieval_augmented (Python)
  - Adds lightweight retrieved context:
    - Docstrings from the given code (module, functions, classes).
    - Short documentation snippets for a small allowlist of stdlib imports (e.g., `math`, `re`, `json`).
  - Benefits: more accurate references to APIs used in the snippet.
- execution_trace (Python)
  - Runs the snippet in a safe sandboxed subprocess with strict limits (CPU time, memory, cleared env).
  - Includes `stdout`/`stderr` in the prompt. Use for small, deterministic code only.
  - Safety: resource limits applied, but never run untrusted code.

Advanced options:

- enhanced_rag (Python)
  - Hybrid retrieval (FAISS + BM25), cross-encoder reranking, and MMR diversity.
  - Tunables: `top_k`, `use_reranker`, `mmr_lambda`, `similarity_threshold`.
  - Best for longer or unfamiliar codebases.

Examples:

```bash
# CLI
cx-explain --prompt-strategy ast_augmented "def add(a,b): return a+b"

# API
curl -X POST http://localhost:8000/explain \
  -H 'Content-Type: application/json' \
  -d '{"code": "print(1+2)", "strategy": "execution_trace"}'

# Eval
code-explainer eval -c configs/default.yaml --prompt-strategy retrieval_augmented --max-samples 5
```
