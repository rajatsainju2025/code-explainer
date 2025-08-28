# Research Agenda: Open Evaluations and LLM Code Understanding (2025 H2)

This document outlines proposed research directions aligned with recent advances in open evaluations and LLM evals.

## Themes

1. Robustness & Reliability in Code Explanation
2. Retrieval-augmented evaluation and anti-overfitting checks
3. Safety-aware explanation and redaction fidelity
4. Grounded explanations with provenance scoring
5. Multilingual and cross-language code understanding

## Experiments

- Counterfactual evaluations (mutated code) to test causal understanding
- Adversarial prompts for prompt-robustness
- Provenance-aware scoring using citation precision/recall
- Hard negative mining for retrieval and re-ranking
- Judging models with rubric and pairwise preference learning

## Milestones

- M1: Eval v1 with counterfactuals and provenance (2 weeks)
- M2: Safety and redaction fidelity evals (2 weeks)
- M3: Cross-language pilot (Python↔Java) (3 weeks)
- M4: Leaderboard and model cards with eval-ready artefacts (2 weeks)

## Reading List (selected)

- Open LLM Leaderboard & Eval Cards (2024–2025)
- HELM 2.0: Holistic Evaluation of Language Models
- LLM-as-a-Judge and Reliability Analyses
- RAG Triad: Retrieval, Generation, Evaluation Best Practices
- CodeBench and CodeExplain Task Suites (community)
