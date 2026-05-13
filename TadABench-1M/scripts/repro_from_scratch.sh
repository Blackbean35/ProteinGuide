#!/usr/bin/env bash
set -euo pipefail

# Full reproducibility: clean, install, run.

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] 'uv' is required for this script. See https://docs.astral.sh/uv/"
  exit 1
fi

# Offline-only: ensure local data/ exists instead of using any network dataset ID
for d in data/all.AA.train data/all.AA.val data/all.AA.test; do
  if [[ ! -d "$d" ]]; then
    echo "[error] missing required dataset directory: $d"
    echo "Place the offline dataset under data/ before running."
    exit 1
  fi
done

bash scripts/clean.sh
uv sync

echo "[repro] Checking environment..."
uv run python scripts/check_env.py

echo "[repro] Running configs..."
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_ESM2-35M.py
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_ESMC-300M.py
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_NT-50M.py

echo "[repro] Completed."
