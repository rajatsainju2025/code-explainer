# Plan Review and Critique (Aug–Sep 2025)

This document reviews the existing contribution plan, research agenda, and RFCs, highlighting strengths, gaps, and concrete improvements.

## Summary

- Strong foundation: tests, CI, contamination checks, provenance, robustness, preference/LLM-judge flows.
- Gaps: external validity, judge reliability, data governance, multi-repo/codebase scale, cost tracking, and ablation discipline.
- Recommendation: pivot to “Evaluation-first Code RAG” with transparent, reproducible benchmarks and judge reliability assessments.

## What’s solid

- Incremental PR cadence and issues hygiene in the 10-day plan; good for community onboarding.
- Open evaluations: contamination detection utilities, provenance metrics, self-consistency hooks.
- RFC Phase 2 targets the right axes: counterfactuals, provenance, safety/redaction.
- Docs footprint (installation, quickstart, advanced evals) is above-average for early-stage research repos.

## Key gaps and risks

1) Judge reliability and variance
- No systematic judge calibration, inter-judge agreement, or sensitivity analysis.
- Missing explicit safeguards against position bias and verbosity bias in pairwise prefs.

2) External validity and overfitting
- Heavy focus on internal datasets; limited cross-benchmark validation (SWE-bench-lite, MBPP+, HumanEval variants, CodeXGLUE subsets, Realistic Code QA).
- Limited holdout/rotation policy for nightly vs. weekly evals.

3) RAG auditability and provenance
- Provenance precision/recall exists but lacks “coverage” and “hallucination penalty” formulations.
- No retrieval heatmaps or per-example “explanation-to-source” alignment reports.

4) Dataset governance and ethics
- No artifact lineage (who added, from where, license, timestamp), license compliance, or PII rules per split.
- Missing contribution checklist for datasets (intake form + automated checks).

5) Cost, latency, and energy
- Benchmarks focus on time/memory; no $/query, tokens/query, or energy proxies.
- No throughput SLOs or service-level gates in CI for regressions.

6) Reproducibility and ablations
- Few scripted ablations (k in RAG, reranker on/off, chunking strategy, judge model, temperature, SC samples).
- Lacking seeds and version pinning for all eval artifacts.

## Suggested changes (short-term)

- Judge reliability:
  - Add calibration set, report inter-judge agreement (Cohen’s κ/Fleiss’ κ) and bootstrap CIs.
  - Debias pairwise comparisons (swap A/B order, normalize verbosity).
- External benchmarks:
  - Wire adapters for SWE-bench-lite, MBPP+, HumanEval+, and a small CodeXGLUE extract.
  - Nightly small subset, weekly full sweep; maintain historical trend plots.
- RAG auditability:
  - Add coverage = cited_tokens / relevant_tokens; hallucination penalty for uncited claims.
  - Generate per-example “provenance cards” (HTML/Markdown) with highlighted spans.
- Data governance:
  - Dataset intake form (YAML) with source, license, timestamp; validate in CI.
  - PII/secrets scanners on PRs that touch datasets.
- Cost/latency/energy:
  - Track tokens, requests, and $/run; add optional energy proxy via CPU/GPU time to Joules estimator.
- Reproducibility/ablations:
  - Build ablation matrix CLI to sweep strategies and produce compact CSV/plots.

## Suggested changes (medium-term)

- Human-in-the-loop eval UI for adjudication and gold creation.
- Counterfactual generation using AST and semantic invariants; publish small open set.
- “Eval Cards” per model and dataset: scope, risks, contamination summary, judge reliability.
- Leaderboard-lite: local static site generated from results/ directory.

## References to align with (non-exhaustive)

- HELM 2.0 (holistic evals), AlpacaEval 2.0 (pairwise prefs), Arena-Hard 2024, RewardBench 2024.
- SWE-bench / SWE-bench-lite (software tasks), HumanEval/MBPP+ (code gen/testing).
- RAG best practices on attribution and grounding from 2023–2024 literature.

---

See `NEXT_PHASE_ROADMAP.md` for an actionable breakdown and milestones.
