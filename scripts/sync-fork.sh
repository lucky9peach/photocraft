#!/bin/sh
# Tri-sync: upstream -> fork main -> current feature branch.
# Usage: ./scripts/sync-fork.sh
# Convention: main always mirrors upstream; all beauty work lives on beauty/*.
set -eu
cd "$(dirname "$0")/.."
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree dirty. Commit or stash first." >&2
  exit 1
fi
git fetch upstream
git fetch origin
cur=$(git branch --show-current)
git checkout -q main
if git merge --ff-only upstream/main; then
  echo "main fast-forwarded."
else
  echo "main is NOT a clean fast-forward. Resolve manually, then re-run." >&2
  git checkout -q "$cur"
  exit 1
fi
git push origin main
if [ "$cur" != "main" ]; then
  git checkout -q "$cur"
  echo "Rebasing $cur onto main..."
  git rebase main
  echo "Done. Push with: git push --force-with-lease origin $cur"
fi
