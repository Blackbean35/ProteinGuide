#!/usr/bin/env bash
set -euo pipefail

# Clean project artifacts to enable a fresh rerun.
# This does NOT touch global caches or your package manager's environment.

echo "[clean] Removing checkpoints and caches..."
rm -rf ckpt 2>/dev/null || true
rm -rf wandb 2>/dev/null || true

echo "[clean] Removing Python bytecode..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "[clean] Done."

if [[ "${1:-}" == "--all" ]]; then
  echo "[clean] Removing local virtual env (.venv) as requested..."
  rm -rf .venv 2>/dev/null || true
  echo "[clean] Done --all."
fi

