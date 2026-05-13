# ProteinGuide

**On-the-fly property-guided protein sequence generation for genome editor engineering.**

ProteinGuide is a two-round pipeline that combines pre-trained generative models (ESM3, ProteinMPNN) with lightweight property predictors to iteratively design high-activity protein variants. The primary application in this repository is **TadA (adenosine base editor) engineering**, using the TadABench-1M dataset to train the predictor.

---

## Overview

```
Round 1  →  Unguided library generation (diverse, broad exploration)
              ↓  [wet-lab assay OR use TadABench-1M directly]
Round 1.5 → Predictor training (sequence → activity regression/classification)
              ↓
Round 2  →  Guided library generation (enriched for high-activity variants)
```

- **No experimental data?** Use TadABench-1M (included under `TadABench-1M/data/`) to pre-train the predictor and skip straight to guided generation.
- **Have experimental data?** Train the predictor on your own assay results (`scripts/02_train_predictor.py`) and run guided generation.

---

## Repository Structure

```
26_ProteinGuide/
│
├── scripts/                        # Main entry-point scripts (run these)
│   ├── 01_generate_library.py      # Round 1: unguided generation
│   ├── 02_train_predictor.py       # Round 1.5: predictor training
│   └── 03_guided_generation.py     # Round 2: guided generation
│
├── protein_guide/                  # Core library
│   ├── models/
│   │   ├── esm3_model.py           # ESM3 generative model wrapper
│   │   └── proteinmpnn_model.py    # ProteinMPNN wrapper
│   ├── guidance/
│   │   ├── deg_sampler.py          # DEG algorithm (Discrete-time Exact Guidance)
│   │   └── tag_sampler.py          # TAG algorithm (fast, approximate)
│   ├── predictors/
│   │   ├── linear_predictor.py     # LinearPairwisePredictor (main predictor)
│   │   └── base_predictor.py       # Abstract base class
│   ├── data/
│   │   ├── sequence_utils.py       # Encoding, FASTA I/O, diversity metrics
│   │   └── structure_utils.py      # PDB loading
│   └── utils/
│       └── masking.py              # Masked sequence utilities
│
├── configs/
│   └── default_config.yaml         # Template YAML config (copy & edit for your target)
│
├── TadABench-1M/                   # Pre-built dataset for TadA predictor training
│   ├── data/
│   │   ├── all.AA.train/           # 256,429 samples (Arrow format)
│   │   ├── all.AA.val/             # 45,208 samples
│   │   ├── all.AA.test/            # 108,232 samples
│   │   ├── all.DNA.train/          # 729,302 samples (codon-level)
│   │   ├── all.DNA.val/
│   │   └── all.DNA.test/
│   ├── config/                     # ESM2/ESMC/NT backbone configs for TadABench
│   └── src/                        # MLP-head training pipeline for TadABench
│
└── requirements.txt
```

---

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. (Optional) ProteinMPNN — only needed if using proteinmpnn model
git clone https://github.com/dauparas/ProteinMPNN.git
```

**Key dependencies:**
- `torch >= 2.0` + CUDA GPU (strongly recommended; CPU fallback supported but slow)
- `esm` — ESM3 from EvolutionaryScale
- `biopython`, `biotite` — PDB structure handling
- `datasets` — for loading TadABench-1M Arrow files

---

## Quick Start: TadA Engineering with TadABench-1M

This is the primary use case. TadABench-1M provides ~410K (AA) TadA variant sequences paired with base editing activity values (`Value` field), allowing predictor training without any new wet-lab experiments.

### Step 1: Train predictor from TadABench-1M

The TadABench-1M dataset uses **HuggingFace `datasets` Arrow format** and is already split into train/val/test. To train the MLP-head predictor bundled with TadABench-1M:

```bash
cd TadABench-1M

# Using ESM2-35M backbone (fastest)
python scripts/run.py --cfg_path config/NB1M_ood_MLP_ESM2-35M.py

# Using ESMC-300M backbone (better accuracy)
python scripts/run.py --cfg_path config/NB1M_ood_MLP_ESMC-300M.py

# Using Nucleotide Transformer (DNA-level)
python scripts/run.py --cfg_path config/NB1M_ood_MLP_NT-50M.py
```

Dataset fields:
| Field | Type | Description |
|---|---|---|
| `Sequence` | `str` | TadA variant sequence (167 AA fixed length) |
| `Value` | `float64` | Base editing activity (raw, not normalized; range ~0.001–14) |

> **Note:** The predictor is evaluated by Spearman correlation (`sp`), Recall@10%, and NDCG@10% — rank-based metrics are preferred since the absolute scale of `Value` is not fully characterized.

### Step 2: Run guided generation (Round 2) with trained predictor

```bash
python scripts/03_guided_generation.py \
    --predictor models/tada_predictor.pt \
    --pdb structures/TadA8e.pdb \
    --wt_sequence MSEVEFSHEYWMRHALTLAKRARDEREVPVGAVLVLNNRVIGEGWNRAIGLHDPTAHAEIMALRQGGLVMQNYRLIDATLYVTFEPCVMCAGAMIHSRIGRVVFGVRNAKTGAAGSLMDVLHYPGMNHRVEITEGILADECAALLCDFYRMPRQVFNAQKKAQSSTD \
    --designable_start 0 --designable_end 167 \
    --fixed_positions 56,57,60 \
    --model esm3 \
    --algorithm deg \
    --gamma 100 \
    --n_samples 500 \
    --output output/tada_guided.fasta
```

---

## Script Reference

### `scripts/01_generate_library.py` — Round 1: Unguided Generation

Generates a diverse library **without** guidance, sweeping over multiple `wt_weight` values to control how conservative/diverse the library is.

```bash
python scripts/01_generate_library.py \
    --config configs/default_config.yaml
    # OR override individually:
    --wt_sequence <AA_SEQ> \
    --pdb <PDB_PATH> \
    --designable_start 0 --designable_end 167 \
    --fixed_positions 56,57,60 \
    --model esm3 \
    --n_samples 1000 \
    --temperature 0.5 \
    --wt_weights "0,0.5,1.0,1.5,2.0,2.5,3.0,3.5" \
    --output output/round1_library.fasta
```

**Key parameters:**
| Parameter | Default | Meaning |
|---|---|---|
| `--wt_weights` | `"0,0.5,...,3.5"` | Controls diversity. 0 = fully random; 3.5 = close to WT |
| `--temperature` | `0.5` | Sampling temperature; lower = more deterministic |
| `--fixed_positions` | `""` | Comma-separated 0-indexed positions locked to WT (e.g. active site) |
| `--model` | `esm3` | `esm3` or `proteinmpnn` |
| `--n_samples` | `1000` | Total sequences (split evenly across `wt_weights`) |

**Output:** `output/round1_library.fasta` + `output/round1_library.csv` (with per-sequence mutation counts and wt_weight metadata)

---

### `scripts/02_train_predictor.py` — Round 1.5: Predictor Training

Trains a `LinearPairwisePredictor` on experimental assay data (CSV). Runs in two phases:
1. **Clean training** — standard supervised learning on complete sequences
2. **Noisy training** — masked input augmentation, making the predictor robust to partially-decoded sequences (required for guidance)

```bash
python scripts/02_train_predictor.py \
    --data data/round1_results.csv \
    --sequence_column sequence \
    --label_column log_enrichment \
    --wt_sequence <AA_SEQ> \
    --designable_start 0 --designable_end 167 \
    --threshold 0.0 \
    --filter_range "-0.25,0.25" \
    --clean_epochs 400 --noisy_epochs 400 \
    --output models/predictor.pt
```

**Key parameters:**
| Parameter | Default | Meaning |
|---|---|---|
| `--threshold` | `0.0` | Converts regression labels to binary (>= threshold = positive) |
| `--filter_range` | `None` | Exclude sequences in ambiguous zone, e.g. `"-0.25,0.25"` |
| `--reg_lambda` | `10.0` | L2 regularization on pairwise terms (increase to prevent overfitting on small data) |
| `--n_cv_splits` | `5` | Cross-validation folds for AUROC evaluation |

**Output:** `models/predictor.pt` (loadable via `predictor.load()`)

---

### `scripts/03_guided_generation.py` — Round 2: Guided Generation

Combines a generative model with a trained predictor using either DEG or TAG guidance.

```bash
python scripts/03_guided_generation.py \
    --predictor models/predictor.pt \
    --pdb <PDB_PATH> \
    --wt_sequence <AA_SEQ> \
    --designable_start 0 --designable_end 167 \
    --model esm3 \
    --algorithm deg \
    --gamma 100 \
    --n_samples 500 \
    --output output/round2_guided.fasta
```

**Guidance algorithms:**
| Algorithm | Description |
|---|---|
| `deg` | **Discrete-time Exact Guidance** — evaluates all 20 amino acids at each position; exact Bayesian inference; recommended |
| `tag` | **Twisted Approximate Guidance** — faster but approximate; use when DEG is too slow |

**`--gamma` (guidance strength):**
- `1.0` = pure Bayes' rule (light guidance)
- `100` = strong guidance (recommended starting point)
- Higher = more high-scoring sequences, but less diversity

**Output:** `output/round2_guided.fasta` + `output/round2_guided.csv` (with `predicted_score`, `n_mutations` per sequence)

---

## YAML Config (configs/default_config.yaml)

All scripts accept `--config <yaml_file>` with CLI flags overriding YAML values. Edit `configs/default_config.yaml` for your target:

```yaml
project:
  wt_sequence: "MSEVEFSHEYWM..."   # Full WT amino acid sequence
  pdb_path: "structures/target.pdb"
  n_chains: 1                       # 1 = monomer, 2 = homodimer

design:
  designable_start: 0               # 0-indexed, inclusive
  designable_end: 167               # 0-indexed, exclusive
  fixed_positions: [56, 57, 60]     # Lock these to WT

generation:
  model: "esm3"
  n_samples: 1000
  temperature: 0.5
  wt_weight: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

guidance:
  algorithm: "deg"
  gamma: 100
  n_samples: 500
```

---

## Core Concepts

### Predictor: `LinearPairwisePredictor`
A linear model over **single-site** and **pairwise** amino acid features. It is lightweight, interpretable, and designed for the small-data regime (hundreds to thousands of sequences). The noisy training phase (masked inputs) makes it usable during autoregressive decoding where some positions are still masked.

### DEG (Discrete-time Exact Guidance)
At each autoregressive decoding step, for position `i`:
```
p_guided(s) ∝ p_gen(s | x_decoded) · predictor(x with s at position i) ^ γ
```
All 20 amino acids are evaluated per position, so it is exact but O(20 × L × n_samples) in predictor calls.

### `wt_weight` parameter
Adds a log-linear bias toward the wild-type sequence during generative model sampling:
- `wt_weight = 0` → no bias (most diverse)
- `wt_weight = 3.5` → strong WT bias (conservative, few mutations)

Sweeping across multiple `wt_weight` values in Round 1 generates a library that spans the exploration-exploitation spectrum.

---

## TadABench-1M Details

| Split | Samples | File size |
|---|---|---|
| AA train | 256,429 | ~44 MB |
| AA val | 45,208 | ~8 MB |
| AA test | 108,232 | ~19 MB |
| DNA train | 729,302 | ~357 MB (via Git LFS) |
| DNA val | 148,014 | ~73 MB (via Git LFS) |
| DNA test | 149,884 | ~72 MB (via Git LFS) |

All Arrow files are stored via **Git LFS**. When cloning, ensure Git LFS is installed:
```bash
git lfs install
git clone https://github.com/Blackbean35/ProteinGuide.git
```

---

## Notes for Codex

- All scripts use `sys.path.insert(0, ...)` so they can be run from any directory.
- The `protein_guide/` package has no `setup.py`; it is imported directly via path manipulation.
- GPU is auto-detected; scripts fall back to CPU if CUDA is unavailable (slower).
- `--fixed_positions` uses **0-indexed** positions throughout.
- Output directories are auto-created if they don't exist.
