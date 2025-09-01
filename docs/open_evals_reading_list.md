# Open Evals and LLM Evaluation: Recent Readings (curated)

A living list to ground our design choices. Please PR updates with brief takeaways.

- HELM 2.0: Holistic Evaluation of Language Models — comprehensive multi-axis evaluation; emphasizes reliability and coverage.
- AlpacaEval 2.0 & Arena-Hard 2024 — pairwise preference evaluation, position-bias mitigation, and robustness of rankings.
- RewardBench (2024) — evaluating reward models; insights for LLM-as-a-judge reliability.
- SWE-bench / SWE-bench-lite — task-oriented software engineering benchmarks, realistic code contexts.
- MBPP+ / HumanEval+ — classic code generation tasks with augmentations; watch for prompt leakage.
- RAG evaluation best practices (2023–2024) — attribution and grounding metrics; provenance coverage and hallucination penalties.
- Open LLM Leaderboards & Eval Cards (2024–2025) — trends in transparent reporting and contamination policies.

Design implications
- Report error bars and agreement (κ), not just point metrics.
- Include contamination checks and data lineage.
- Preference evals must randomize A/B order and normalize verbosity.
- RAG needs provenance coverage and hallucination penalties, plus per-example cards.
