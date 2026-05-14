#!/usr/bin/env python3
"""
Round 2: Guided Library Generation

Generates sequences guided by a trained property predictor using
ProteinGuide's DEG or TAG algorithm. Requires a pre-trained predictor
(from Round 1.5 or user-provided).

Usage (TadA engineering with TadABench-1M predictor):
    # 1. Train predictor first:
    python scripts/00_train_tada_predictor.py

    # 2. Run guided generation:
    python scripts/03_guided_generation.py \
        --config configs/tada_config.yaml \
        --predictor models/tada_esm2_predictor.pt \
        --predictor_type esm2 \
        --algorithm deg --gamma 100 --n_samples 500

Usage (custom predictor):
    python scripts/03_guided_generation.py \
        --predictor models/predictor.pt \
        --predictor_type linear_pairwise \
        --pdb structures/my_editor.pdb \
        --wt_sequence MSEVEFSHE... \
        --designable_start 0 --designable_end 167 \
        --fixed_positions 22,56,58,96 \
        --model esm3 --algorithm deg --gamma 100 --n_samples 500 \
        --output output/round2_guided.fasta
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_guide.models.esm3_model import ESM3Model
from protein_guide.data.structure_utils import load_pdb_structure
from protein_guide.data.sequence_utils import (
    sequences_to_fasta, mutation_count, compute_diversity
)
from protein_guide.predictors.linear_predictor import LinearPairwisePredictor
from protein_guide.guidance.deg_sampler import DEGSampler
from protein_guide.guidance.tag_sampler import TAGSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Guided sequence generation")
    
    parser.add_argument("--config", type=str, help="YAML config file")
    
    # Predictor
    parser.add_argument("--predictor", type=str, default=None,
                        help="Path to trained predictor (.pt file). "
                             "If not set, reads from --config predictor.checkpoint")
    parser.add_argument("--predictor_type", type=str, default=None,
                        choices=["linear_pairwise", "esm2"],
                        help="Predictor type: 'esm2' (TadABench-1M) or 'linear_pairwise'")
    parser.add_argument("--backbone", type=str, default=None,
                        help="ESM2 backbone (only for --predictor_type esm2). "
                             "Defaults to value in checkpoint or config.")
    
    # Structure & Sequence
    # These can be set via --config YAML (tada_config.yaml has them pre-filled)
    parser.add_argument("--pdb", type=str, default=None,
                        help="Path to PDB structure file. Or set via YAML config.")
    parser.add_argument("--chain_id", type=str, default=None)
    parser.add_argument("--wt_sequence", type=str, default=None,
                        help="Wild-type amino acid sequence. Or set via YAML config.")
    parser.add_argument("--designable_start", type=int, default=None,
                        help="Start of designable region (0-indexed). Or set via YAML.")
    parser.add_argument("--designable_end", type=int, default=None,
                        help="End of designable region (0-indexed, exclusive). Or set via YAML.")
    parser.add_argument("--fixed_positions", type=str, default=None,
                        help="Comma-separated 0-indexed positions to keep at WT. Or set via YAML.")
    
    # Model
    parser.add_argument("--model", type=str, default="esm3",
                        choices=["esm3", "proteinmpnn"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_chains", type=int, default=1)
    
    # Guidance
    parser.add_argument("--algorithm", type=str, default="deg",
                        choices=["deg", "tag"])
    parser.add_argument("--gamma", type=float, default=100.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.01, help="TAG step size")
    parser.add_argument("--wt_weight", type=float, default=0.0)
    
    # Sampling
    parser.add_argument("--n_samples", type=int, default=500)
    
    # Output
    parser.add_argument("--output", type=str, default="output/round2_guided.fasta")
    parser.add_argument("--output_csv", type=str, default="output/round2_guided.csv")
    
    return parser.parse_args()


def _load_config_yaml(config_path):
    """Load YAML config and return as a simple namespace-like dict."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Merge YAML config (CLI args take priority)
    # ------------------------------------------------------------------
    cfg = {}
    if args.config:
        cfg = _load_config_yaml(args.config)

    def get(cli_val, *keys, default=None):
        """Return cli_val if set, else walk cfg dict by keys, else default."""
        if cli_val is not None:
            return cli_val
        node = cfg
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node if node is not None else default

    wt_sequence      = get(args.wt_sequence,       "project", "wt_sequence")
    pdb_path         = get(args.pdb,               "project", "pdb_path")
    chain_id         = get(args.chain_id,          "project", "chain_id")
    n_chains         = get(args.n_chains,          "project", "n_chains",         default=1)
    des_start        = get(args.designable_start,  "design",  "designable_start",  default=0)
    des_end          = get(args.designable_end,    "design",  "designable_end")
    fixed_raw        = get(args.fixed_positions,   "design",  "fixed_positions",   default="")
    model_name       = get(args.model,             "generation", "model",          default="esm3")
    algorithm        = get(args.algorithm,         "guidance", "algorithm",        default="deg")
    gamma            = get(args.gamma,             "guidance", "gamma",            default=100.0)
    temperature      = get(args.temperature,       "guidance", "temperature",      default=0.5)
    dt               = get(args.dt,                "guidance", "dt",               default=0.01)
    wt_weight        = get(args.wt_weight,         "guidance", "wt_weight",        default=0.0)
    n_samples        = get(args.n_samples,         "guidance", "n_samples",        default=500)
    predictor_path   = get(args.predictor,         "predictor", "checkpoint")
    predictor_type   = get(args.predictor_type,    "predictor", "type",            default="linear_pairwise")
    backbone         = get(args.backbone,          "predictor", "backbone",        default="facebook/esm2_t12_35M_UR50D")
    output_fasta     = get(args.output,            "output",   "round2_fasta",     default="output/round2_guided.fasta")
    output_csv       = get(args.output_csv,        "output",   "round2_csv",       default="output/round2_guided.csv")

    # Validate required fields
    assert wt_sequence,   "wt_sequence is required (via --wt_sequence or config)"
    assert pdb_path,      "pdb is required (via --pdb or config)"
    assert des_end,       "designable_end is required"
    assert predictor_path, "predictor checkpoint is required (via --predictor or config predictor.checkpoint)"

    # Parse fixed positions
    fixed_positions = []
    if isinstance(fixed_raw, list):
        fixed_positions = [int(x) for x in fixed_raw]
    elif isinstance(fixed_raw, str) and fixed_raw:
        fixed_positions = [int(x) for x in fixed_raw.split(",")]

    designable_positions = list(range(int(des_start), int(des_end)))

    logger.info("=" * 60)
    logger.info("ProteinGuide — Round 2: Guided Library Generation")
    logger.info("=" * 60)
    logger.info(f"WT length     : {len(wt_sequence)}")
    logger.info(f"Designable    : {len(designable_positions)} positions")
    logger.info(f"Fixed         : {fixed_positions}")
    logger.info(f"Algorithm     : {algorithm.upper()}, γ={gamma}")
    logger.info(f"Predictor type: {predictor_type}")
    logger.info(f"Predictor path: {predictor_path}")

    # Load structure
    logger.info(f"Loading structure: {pdb_path}")
    structure = load_pdb_structure(pdb_path, chain_id)

    # ------------------------------------------------------------------
    # Load predictor
    # ------------------------------------------------------------------
    logger.info(f"Loading predictor ({predictor_type})...")
    if predictor_type == "esm2":
        from protein_guide.predictors.esm2_predictor import ESM2MLPPredictor
        predictor = ESM2MLPPredictor.from_checkpoint(
            checkpoint_path=predictor_path,
            device=args.device,
        )
    else:  # linear_pairwise
        predictor = LinearPairwisePredictor(
            seq_length=len(wt_sequence),
            designable_positions=designable_positions,
            device=args.device if args.device != "cuda" else "cpu",
        )
        predictor.load(predictor_path)
    
    # Load generative model
    logger.info(f"Loading {model_name} model...")
    if model_name == "esm3":
        gen_model = ESM3Model(device_str=args.device, n_chains=int(n_chains))
    elif model_name == "proteinmpnn":
        from protein_guide.models.proteinmpnn_model import ProteinMPNNModel
        gen_model = ProteinMPNNModel(device_str=args.device, n_chains=int(n_chains))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Set up sampler
    if algorithm == "deg":
        sampler = DEGSampler(
            generative_model=gen_model,
            predictor=predictor,
            gamma=float(gamma),
            temperature=float(temperature),
            wt_weight=float(wt_weight),
        )
    elif algorithm == "tag":
        sampler = TAGSampler(
            generative_model=gen_model,
            predictor=predictor,
            gamma=float(gamma),
            temperature=float(temperature),
            dt=float(dt),
            wt_weight=float(wt_weight),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Generate
    logger.info(f"\nGenerating {n_samples} guided sequences...")
    results = sampler.sample(
        structure_data=structure,
        wt_sequence=wt_sequence,
        designable_positions=designable_positions,
        fixed_positions=fixed_positions,
        n_samples=int(n_samples),
    )
    
    # Save outputs
    output_dir = Path(output_fasta).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # FASTA
    sequences = [r["sequence"] for r in results]
    names = [f"guided_{i:04d}" for i in range(len(results))]
    scores = [r["predicted_score"] for r in results]
    sequences_to_fasta(sequences, output_fasta, names=names, scores=scores)
    logger.info(f"FASTA saved: {output_fasta}")

    # CSV
    df = pd.DataFrame(results)
    df.insert(0, "name", names)
    df["algorithm"] = algorithm
    df["gamma"] = gamma
    df["model"] = model_name
    df["predictor_type"] = predictor_type
    df.to_csv(output_csv, index=False)
    logger.info(f"CSV saved: {output_csv}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info(f"  Total          : {len(results)} sequences")
    logger.info(f"  Predicted >0.5 : {sum(s > 0.5 for s in scores)}/{len(scores)}")
    logger.info(f"  Mean score     : {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    logger.info(f"  Mean mutations : {np.mean([r['n_mutations'] for r in results]):.1f}")
    if len(sequences) > 1:
        div = compute_diversity(sequences[:50])
        logger.info(f"  Diversity      : {div:.3f}")
    logger.info(f"  Output FASTA   : {output_fasta}")
    logger.info(f"  Output CSV     : {output_csv}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
