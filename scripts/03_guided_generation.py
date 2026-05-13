#!/usr/bin/env python3
"""
Round 2: Guided Library Generation

Generates sequences guided by a trained property predictor using
ProteinGuide's DEG or TAG algorithm. Requires a pre-trained predictor
(from Round 1.5 or user-provided).

Usage:
    python scripts/03_guided_generation.py \
        --predictor models/predictor.pt \
        --pdb structures/my_editor.pdb \
        --wt_sequence MSEVEFSHE... \
        --designable_start 82 --designable_end 167 \
        --fixed_positions 86,87,90 \
        --model esm3 \
        --algorithm deg \
        --gamma 100 \
        --n_samples 500 \
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
    parser.add_argument("--predictor", type=str, required=True,
                        help="Path to trained predictor (.pt file)")
    parser.add_argument("--predictor_type", type=str, default="linear_pairwise")
    
    # Structure & Sequence
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--chain_id", type=str, default=None)
    parser.add_argument("--wt_sequence", type=str, required=True)
    parser.add_argument("--designable_start", type=int, required=True)
    parser.add_argument("--designable_end", type=int, required=True)
    parser.add_argument("--fixed_positions", type=str, default="")
    
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


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("ProteinGuide — Round 2: Guided Library Generation")
    logger.info("=" * 60)
    
    # Parse fixed positions
    fixed_positions = []
    if args.fixed_positions:
        fixed_positions = [int(x) for x in args.fixed_positions.split(",")]
    
    designable_positions = list(range(args.designable_start, args.designable_end))
    
    logger.info(f"WT length: {len(args.wt_sequence)}")
    logger.info(f"Designable: {len(designable_positions)} positions")
    logger.info(f"Fixed: {fixed_positions}")
    logger.info(f"Algorithm: {args.algorithm.upper()}, γ={args.gamma}")
    
    # Load structure
    logger.info(f"Loading structure: {args.pdb}")
    structure = load_pdb_structure(args.pdb, args.chain_id)
    
    # Load predictor
    logger.info(f"Loading predictor: {args.predictor}")
    predictor = LinearPairwisePredictor(
        seq_length=len(args.wt_sequence),
        designable_positions=designable_positions,
        device=args.device if args.device != "cuda" else "cpu",
    )
    predictor.load(args.predictor)
    
    # Load generative model
    logger.info(f"Loading {args.model} model...")
    if args.model == "esm3":
        gen_model = ESM3Model(device_str=args.device, n_chains=args.n_chains)
    elif args.model == "proteinmpnn":
        from protein_guide.models.proteinmpnn_model import ProteinMPNNModel
        gen_model = ProteinMPNNModel(device_str=args.device, n_chains=args.n_chains)
    
    # Set up sampler
    if args.algorithm == "deg":
        sampler = DEGSampler(
            generative_model=gen_model,
            predictor=predictor,
            gamma=args.gamma,
            temperature=args.temperature,
            wt_weight=args.wt_weight,
        )
    elif args.algorithm == "tag":
        sampler = TAGSampler(
            generative_model=gen_model,
            predictor=predictor,
            gamma=args.gamma,
            temperature=args.temperature,
            dt=args.dt,
            wt_weight=args.wt_weight,
        )
    
    # Generate
    logger.info(f"\nGenerating {args.n_samples} guided sequences...")
    results = sampler.sample(
        structure_data=structure,
        wt_sequence=args.wt_sequence,
        designable_positions=designable_positions,
        fixed_positions=fixed_positions,
        n_samples=args.n_samples,
    )
    
    # Save outputs
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FASTA
    sequences = [r["sequence"] for r in results]
    names = [f"guided_{i:04d}" for i in range(len(results))]
    scores = [r["predicted_score"] for r in results]
    sequences_to_fasta(sequences, args.output, names=names, scores=scores)
    logger.info(f"FASTA saved: {args.output}")
    
    # CSV
    df = pd.DataFrame(results)
    df.insert(0, "name", names)
    df["algorithm"] = args.algorithm
    df["gamma"] = args.gamma
    df["model"] = args.model
    df.to_csv(args.output_csv, index=False)
    logger.info(f"CSV saved: {args.output_csv}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info(f"  Total: {len(results)} sequences")
    logger.info(f"  Predicted active (>0.5): {sum(s > 0.5 for s in scores)}/{len(scores)}")
    logger.info(f"  Mean score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    logger.info(f"  Mean mutations: {np.mean([r['n_mutations'] for r in results]):.1f}")
    if len(sequences) > 1:
        div = compute_diversity(sequences[:50])
        logger.info(f"  Diversity (sample): {div:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
