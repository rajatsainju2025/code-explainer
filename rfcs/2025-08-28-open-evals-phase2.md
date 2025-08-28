# RFC: Open Evals Phase 2 — Counterfactuals, Provenance, and Safety

Date: 2025-08-28
Status: Proposed
Authors: @rajatsainju2025

## Summary

Introduce counterfactual tests, provenance-aware scoring, and safety/redaction fidelity metrics to the evaluation suite.

## Motivation

- Ensure explanations remain correct under minimal code mutations
- Reward explanations that cite retrieved sources (retrieval grounding)
- Detect leakage and unsafe outputs (secrets/PII)

## Design

- Counterfactuals: Programmatically mutate AST preserving semantics in decoys
- Provenance: Track retrieved passages; score citation precision/recall
- Safety: Add detectors for PII/secrets and measure redaction fidelity

## Alternatives

- Human-only evals (costly)
- Synthetic-only evals (may lack realism)

## Risks

- Judge bias; mitigate with rubric and pairwise comparisons
- False positives in safety detectors; calibrate thresholds

## Rollout

- Implement behind a feature flag `--phase2`
- Run nightly on a subset; scale to full weekly

## Metrics

- Delta in BLEU/ROUGE/CodeBLEU on counterfactuals
- Citation precision/recall@k
- Redaction precision/recall
