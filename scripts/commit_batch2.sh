#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 6 Trainer compatibility
git add src/code_explainer/trainer.py
if git diff --staged --quiet; then
  echo "No staged changes for trainer.py"
else
  git commit -m "chore(trainer): accept config_path in constructor for compatibility"
fi

# 7 Retriever and FAISS
git add src/code_explainer/retrieval/retriever.py src/code_explainer/retrieval/faiss_index.py
if git diff --staged --quiet; then
  echo "No staged changes for retriever/faiss"
else
  git commit -m "fix(retrieval): expose model_name and make FAISSIndex instance dict-friendly for tests"
fi

# 8 Model loader and security tweaks
git add src/code_explainer/model_loader.py src/code_explainer/security.py
if git diff --staged --quiet; then
  echo "No staged changes for model_loader/security"
else
  git commit -m "fix(core): small model_loader and security tweaks from test iteration"
fi

# 9 Tests: bulk updates
git add tests/* || true
if git diff --staged --quiet; then
  echo "No staged changes for tests"
else
  git commit -m "test: update tests and add integration adjustments after refactor"
fi

# 10 Scripts and misc
git add scripts/run_failing_tests.sh scripts/commit_batch1.sh || true
if git diff --staged --quiet; then
  echo "No staged changes for scripts"
else
  git commit -m "chore(scripts): add test helpers and commit batching scripts"
fi

# Show last 10 commits
git --no-pager log --oneline -n 10
