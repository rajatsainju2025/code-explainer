# 10-Day High-Impact Contribution Plan

Goal: 10–15 meaningful GitHub contributions per day to grow the project and profile. Treat this as a living document—update daily based on progress and community feedback.

Notes
- Prefer small, reviewable PRs (1 logical change/PR). Merge frequently.
- Mix PRs, issues, discussions, workflows, labels/milestones, docs, tests.
- Avoid spam. Every contribution should improve the repo or community.
- Use milestones and a project board to track completion.

Milestones
- v0.2.3 (Docs & Examples)
- v0.3.0 (Eval & CI)
- v0.3.1 (Packaging/PyPI)

Project Board
- Create “Next 10 Days” project (To do / In progress / In review / Done).

---

Day 1 — 2025-08-08 (Docs + Community)
- [ ] PR: Add Discussions link/badge to README and examples link section (done if already merged)
- [ ] PR: Add Colab badge in README (done if already merged)
- [ ] PR: Add examples/README quickstart sections (train, eval, serve) (done if already merged)
- [ ] PR: Add examples/colab_quickstart.ipynb (done if already merged)
- [ ] Issue: Track “Add Colab quickstart notebook” (created)
- [ ] Issue: Track “Add model presets table” (created)
- [ ] Issue: Track “Add more dataset samples + schema doc” (created)
- [ ] Issue: Track “Add Colab test to CI” (created)
- [ ] Issue: Track “Add PyPI publish workflow” (created)
- [ ] Discussion: Start here thread (created)
- [ ] PR: Add CODE_OF_CONDUCT.md
- [ ] PR: Add SECURITY.md
- [ ] PR: Add FUNDING.yml (GitHub Sponsors placeholder)
- [ ] Admin: Create labels (infra, packaging, testing, docs, enhancement)

Target contributions today: 12–15

---

Day 2 — 2025-08-09 (Tests + CI)
- [ ] PR: Add unit tests for cli explain/explain-file happy path
- [ ] PR: Add tests for datasets loader and prompt templates
- [ ] PR: Add pytest workflow matrix (3.9–3.12, ubuntu-latest)
- [ ] PR: Add CodeQL security scanning workflow
- [ ] PR: Add Ruff or Flake8 annotations to CI output
- [ ] Issue: Good-first-issue to write tests for trainer small config
- [ ] Issue: Good-first-issue to add missing type hints in model/trainer
- [ ] PR: Add coverage.xml generation and upload to artifact
- [ ] PR: Add coverage badge (local placeholder) to README
- [ ] Discussion: Testing strategy thread

Target contributions: 10–12

---

Day 3 — 2025-08-10 (Examples + Data)
- [ ] PR: Expand data/train.json to +20 curated samples (tiny)
- [ ] PR: Expand data/eval.json and data/test.json with +10 each
- [ ] PR: Add data/README.md with schema, guidelines
- [ ] PR: Add examples/preset_switching.md quick guide
- [ ] PR: Add examples/eval_report_template.md
- [ ] Issue: Call for community samples (Discussion link)
- [ ] PR: Add script scripts/validate_dataset.py (schema validation)
- [ ] PR: Add dataset validation step in CI
- [ ] PR: Update README examples section with links
- [ ] Discussion: Data curation best practices

Target contributions: 10–12

---

Day 4 — 2025-08-11 (Eval + Metrics)
- [ ] PR: Add CLI flag to save eval predictions to JSONL
- [ ] PR: Add confusion report summary (basic) in eval output
- [ ] PR: Add aggregate metrics table printing with Rich
- [ ] PR: Add --max-samples to eval for fast CI
- [ ] PR: Add examples/eval_results/ with sample outputs
- [ ] Issue: Good-first-issue to add additional metrics (BLEURT optional)
- [ ] PR: Document eval workflow in README
- [ ] PR: Add eval workflow job in CI (nightly)
- [ ] Discussion: Share first eval results, ask for feedback

Target contributions: 10–11

---

Day 5 — 2025-08-12 (Web + UX)
- [ ] PR: Streamlit app polish (layout, examples, sidebar controls)
- [ ] PR: Gradio app: theme, examples, docstrings
- [ ] PR: Add favicon/logo placeholder
- [ ] PR: Add /health and /version endpoints to FastAPI
- [ ] PR: Add Docker Compose for API + UI
- [ ] Issue: Feature request template improvements
- [ ] PR: README screenshots/gifs of UI
- [ ] Discussion: UX feedback thread
- [ ] PR: Docs on deploying to Hugging Face Spaces (optional)
- [ ] PR: Docs on deploying to Render/Fly.io (optional)

Target contributions: 10–12

---

Day 6 — 2025-08-13 (Packaging + Release)
- [ ] PR: Bump version to 0.2.3 and update CHANGELOG.md
- [ ] PR: Add pyproject metadata (done if merged), twine/build docs
- [ ] PR: Add release.yml for PyPI publish (testpypi first)
- [ ] Issue: Track PyPI token secret setup
- [ ] PR: Build sdist+wheel check workflow (on PR)
- [ ] PR: Add CITATION.cff
- [ ] PR: Add classifiers/badges (PyPI, version)
- [ ] Discussion: Release checklist thread
- [ ] Tag: v0.2.3 after merge (release notes)

Target contributions: 10–12

---

Day 7 — 2025-08-14 (Code Quality)
- [ ] PR: Enable Ruff (or keep Flake8) with clear rules
- [ ] PR: mypy strict on src/code_explainer/* (incremental)
- [ ] PR: Fix typing in model.py (tokenizer types, Optional guards)
- [ ] PR: Add pre-commit hooks for nbstripout on notebooks
- [ ] PR: Add commit message lint (conventional commits)
- [ ] Issue: Good-first-issue for typing in data and metrics modules
- [ ] PR: Docs: style guide for PRs
- [ ] Discussion: Type coverage goals

Target contributions: 10–11

---

Day 8 — 2025-08-15 (Automation + Bots)
- [ ] PR: Add Dependabot config (pip, weekly)
- [ ] PR: Add stale bot config for issues/PRs
- [ ] PR: Add auto-assign reviewers CODEOWNERS
- [ ] PR: Add issue forms (bug, feature) with YAML forms
- [ ] Issue: Invite community to review open PRs
- [ ] PR: Add GitHub Project board (docs) + link in README
- [ ] PR: Add governance doc (maintainer guidelines)
- [ ] Discussion: Roadmap priorities poll

Target contributions: 10–12

---

Day 9 — 2025-08-16 (Models + Presets)
- [ ] PR: Add Llama.cpp or small GGUF preset doc (docs-only)
- [ ] PR: Add guidance for 8-bit/4-bit loading, MPS notes
- [ ] PR: Add preset configs for gpt2-medium and codet5p-220m (docs if heavy)
- [ ] PR: Add model card in docs/models.md
- [ ] PR: Add benchmark script in benchmarks/run_bench.py
- [ ] Issue: Good-first-issue to run and record local benchmarks
- [ ] PR: README table linking new presets
- [ ] Discussion: Model roadmap thread

Target contributions: 10–12

---

Day 10 — 2025-08-17 (Community + Polish)
- [ ] PR: Add CONTRIBUTING quickstart checklist (if not merged)
- [ ] PR: Add first-timers-only labels on selected issues
- [ ] PR: Add screenshots of before/after contributions this sprint
- [ ] PR: Add FAQ to README
- [ ] Issue: Open call for maintainers/co-maintainers
- [ ] Discussion: Sprint retrospective & next steps
- [ ] Tag: v0.3.0 (Docs + Eval + CI) if scope completed
- [ ] Close finished milestone(s)
- [ ] Curate “good first issues” for next week

Target contributions: 10–12

---

Daily Execution Tips
- Time-box work into 3–5 PRs in the morning, 3–5 PRs afternoon; sprinkle issues/discussions across the day.
- Always link PRs to issues; use small diffs and good titles.
- Keep CI green; avoid large refactors in a single PR.

Tracking
- Update this plan daily with checkmarks and notes.
- Adjust counts if some items roll over; keep daily target 10–15.
