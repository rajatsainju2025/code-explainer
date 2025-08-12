# Strategy Examples

```bash
# Vanilla
cx-explain "print('hello world')"

# AST-augmented
cx-explain --prompt-strategy ast_augmented "def add(a,b): return a+b"

# Retrieval-augmented
cx-explain --prompt-strategy retrieval_augmented "import math\nprint(math.sqrt(16))"

# Execution trace (safe, small snippets only)
cx-explain --prompt-strategy execution_trace "print(sum([1,2,3]))"

# Evaluate with a strategy
code-explainer eval --prompt-strategy ast_augmented --max-samples 5

# A/B compare
python scripts/ab_compare_strategies.py --max-samples 5 --strategies vanilla ast_augmented retrieval_augmented
```
