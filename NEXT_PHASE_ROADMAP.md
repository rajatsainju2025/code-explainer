# Next Phase Roadmap (Q4 2025)

This roadmap operationalizes the critique into deliverables with milestones and owners.

## Objectives

- Evaluation-first: trustworthy, reproducible, and explainable evaluation for Code RAG.
- Broaden validity: multiple public benchmarks and realistic internal tasks.
- Governance and ethics: data lineage, licenses, redaction fidelity.

## Milestones

M1 (2 weeks): Judge reliability + reporting
- Inter-judge agreement (κ), bootstrap CIs
- A/B order randomization; verbosity normalization
- CLI: `cx eval-judge-reliability` producing JSON + markdown

M2 (2 weeks): External benchmarks integration
- Adapters: SWE-bench-lite, HumanEval+, MBPP+, CodeXGLUE-small
- Nightly (10%) vs. weekly (100%) suites; trend charts

M3 (2 weeks): RAG auditability and provenance cards
- Coverage metric, hallucination penalty
- Script to emit per-example provenance cards (.md/.html)

M4 (2 weeks): Dataset governance + CI
- Intake YAML + validator
- PII/secrets scan on PRs touching data/

M5 (2 weeks): Ablations + cost/latency tracking
- Ablation matrix CLI (strategies × k × reranker × temperature × SC)
- Cost/tokens recorder; throughput SLO check

## Workstreams

- Eval Core, Data Governance, Benchmarks, Tooling & CI, Docs & Community.

## Risks

- API costs for judges; mitigate with caching and small calibration sets.
- Benchmark drift; pin versions and snapshot data.

## Success Criteria

- Public reports reproducible from a single command.
- Clear, stable metrics with error bars and reliability notes.
