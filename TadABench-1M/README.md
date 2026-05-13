# Overview

Anonymized implementation of an MLP head trained on a large-scale biological sequence dataset. This package is prepared for double‑blind peer review and contains no author or affiliation identifiers.

The code supports multiple backbone encoders (ESM2/ESM3/Nucleotide Transformer) and a simple MLP head for regression tasks. It includes deterministic seeding, step‑by‑step setup, clean re‑run scripts, and troubleshooting guidance.

# Requirements

- Python `>= 3.11`
- GPU with CUDA (recommended) or CPU fallback
- Sufficient memory for backbone models and dataset loading
- Package manager: either `uv` (preferred) or `pip`

Note: Backbones can be large; GPU is strongly recommended. CPU fallback is supported but substantially slower.

# Dataset (Included Offline)
Note that we have prepared the data already. You can skip this step.

- The review artifact includes an offline dataset under `data/` with splits saved via `datasets.save_to_disk`:
  - `data/all.AA.train`, `data/all.AA.val`, `data/all.AA.test`
  - Alternatively for DNA: `data/all.DNA.train`, etc.
- The code automatically prefers `data/` and will not download anything.
  

# Quickstart (uv)

From a fresh clone or archive extraction:

```bash
# 1) Optional: start from a clean slate
bash scripts/clean.sh

# 2) Install dependencies
uv sync

# 3) Verify environment
uv run python scripts/check_env.py

# 4) Run training with provided configs (uses local data/)
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_ESM2-35M.py
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_ESMC-300M.py
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_NT-50M.py
```

If you encounter issues, see Troubleshooting and Re‑run From Scratch below.

# Quickstart (Python venv)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync

python scripts/check_env.py

python scripts/run.py --cfg_path config/NB1M_ood_MLP_ESM2-35M.py
python scripts/run.py --cfg_path config/NB1M_ood_MLP_ESMC-300M.py
python scripts/run.py --cfg_path config/NB1M_ood_MLP_NT-50M.py
```

Alternatively, if you do not use `uv`, install with `pip` following the dependencies listed in `pyproject.toml`.

# Configuration

- Configs live under `config/`.
- Key fields:
  - `embed_name`: Backbone model identifier.
  - `seq_type`: One of `AA` or `DNA`.
  - `length`: Sequence length control for model shapes.
  - `num_epochs`, `batch_size`, `optimizer_type`, `evaluation`, etc.
  - Optional logging with Weights & Biases (disabled by default; keep disabled for anonymous review).

# Determinism and Devices

- The code sets a fixed random seed at runtime.
- Device is auto‑detected; if CUDA is available, CUDA is used; otherwise CPU is used. Training on CPU is slower but supported.

# Notes

- This repository intentionally omits author names, personal links, and non‑essential identifiers for double‑blind review.
- If enabling external logging (e.g., wandb), ensure that project and user identifiers do not de‑anonymize the submission.

# License

Apache 2.0. See `LICENSE` for details.
