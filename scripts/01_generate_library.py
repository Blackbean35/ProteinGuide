#!/usr/bin/env python3
"""
Round 1: Unguided Library Generation

Generates a diverse library of protein sequences using pre-trained
generative models (ESM3 or ProteinMPNN) without guidance.

Usage:
    python scripts/01_generate_library.py --config configs/my_editor.yaml
    
    # Or with command-line overrides:
    python scripts/01_generate_library.py \
        --wt_sequence MSEVEFSHE... \
        --pdb structures/my_editor.pdb \
        --designable_start 82 --designable_end 167 \
        --fixed_positions 86,87,90 \
        --model esm3 \
        --n_samples 1000 \
        --output output/round1_library.fasta
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_guide.models.esm3_model import ESM3Model
from protein_guide.data.structure_utils import load_pdb_structure
from protein_guide.data.sequence_utils import (
    sequences_to_fasta, mutation_count, compute_diversity
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate unguided protein library")
    
    # Config file (optional — overridden by CLI args)
    parser.add_argument("--config", type=str, help="YAML config file path")
    
    # Required inputs
    parser.add_argument("--wt_sequence", type=str, help="Wild-type sequence")
    parser.add_argument("--pdb", type=str, help="PDB structure file path")
    parser.add_argument("--chain_id", type=str, default=None, help="PDB chain ID")
    
    # Design region
    parser.add_argument("--designable_start", type=int, help="Start of designable region (0-indexed)")
    parser.add_argument("--designable_end", type=int, help="End of designable region (0-indexed, exclusive)")
    parser.add_argument("--fixed_positions", type=str, default="",
                        help="Comma-separated fixed positions (0-indexed)")
    
    # Model settings
    parser.add_argument("--model", type=str, default="esm3",
                        choices=["esm3", "proteinmpnn"],
                        help="Generative model to use")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_chains", type=int, default=1,
                        help="Number of chains for homo-oligomers")
    
    # Sampling settings
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--wt_weights", type=str, default="0,0.5,1.0,1.5,2.0,2.5,3.0,3.5",
                        help="Comma-separated wild-type weights to use")
    parser.add_argument("--stochasticity", type=float, default=0.0)
    
    # Output
    parser.add_argument("--output", type=str, default="output/round1_library.fasta")
    parser.add_argument("--output_csv", type=str, default="output/round1_library.csv")
    
    return parser.parse_args()


def load_config(args):
    """Merge YAML config with command-line arguments."""
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    
    # CLI args override config
    result = {
        "wt_sequence": args.wt_sequence or config.get("project", {}).get("wt_sequence"),
        "pdb_path": args.pdb or config.get("project", {}).get("pdb_path"),
        "chain_id": args.chain_id or config.get("project", {}).get("chain_id"),
        "designable_start": args.designable_start or config.get("design", {}).get("designable_start"),
        "designable_end": args.designable_end or config.get("design", {}).get("designable_end"),
        "fixed_positions": [],
        "model": args.model or config.get("generation", {}).get("model", "esm3"),
        "device": args.device,
        "n_chains": args.n_chains or config.get("project", {}).get("n_chains", 1),
        "n_samples": args.n_samples or config.get("generation", {}).get("n_samples", 1000),
        "temperature": args.temperature or config.get("generation", {}).get("temperature", 0.5),
        "stochasticity": args.stochasticity,
        "output": args.output,
        "output_csv": args.output_csv,
    }
    
    # Parse fixed positions
    if args.fixed_positions:
        result["fixed_positions"] = [int(x) for x in args.fixed_positions.split(",")]
    elif "design" in config and "fixed_positions" in config["design"]:
        result["fixed_positions"] = config["design"]["fixed_positions"]
    
    # Parse wt_weights
    if args.wt_weights:
        result["wt_weights"] = [float(x) for x in args.wt_weights.split(",")]
    elif "generation" in config and "wt_weight" in config["generation"]:
        ww = config["generation"]["wt_weight"]
        result["wt_weights"] = ww if isinstance(ww, list) else [ww]
    else:
        result["wt_weights"] = [0.0]
    
    # Validate
    assert result["wt_sequence"], "Wild-type sequence is required"
    assert result["pdb_path"], "PDB file path is required"
    assert result["designable_start"] is not None, "designable_start is required"
    assert result["designable_end"] is not None, "designable_end is required"
    
    return result


def main():
    args = parse_args()
    config = load_config(args)
    
    logger.info("=" * 60)
    logger.info("ProteinGuide — Round 1: Unguided Library Generation")
    logger.info("=" * 60)
    logger.info(f"WT sequence length: {len(config['wt_sequence'])}")
    logger.info(f"Designable region: {config['designable_start']}-{config['designable_end']}")
    logger.info(f"Fixed positions: {config['fixed_positions']}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Samples per wt_weight: {config['n_samples']}")
    
    # Load structure
    logger.info(f"Loading structure from {config['pdb_path']}")
    structure = load_pdb_structure(config["pdb_path"], config["chain_id"])
    logger.info(f"Structure loaded: {len(structure['sequence'])} residues, chain {structure['chain_id']}")
    
    # Set up designable positions
    designable_positions = list(range(config["designable_start"], config["designable_end"]))
    fixed_positions = config["fixed_positions"]
    
    # Load model
    logger.info(f"Initializing {config['model']} model...")
    if config["model"] == "esm3":
        model = ESM3Model(device_str=config["device"], n_chains=config["n_chains"])
    elif config["model"] == "proteinmpnn":
        from protein_guide.models.proteinmpnn_model import ProteinMPNNModel
        model = ProteinMPNNModel(device_str=config["device"], n_chains=config["n_chains"])
    
    # Generate sequences for each wt_weight
    all_sequences = []
    all_metadata = []
    
    for wt_w in config["wt_weights"]:
        n_per_weight = config["n_samples"] // len(config["wt_weights"])
        logger.info(f"\n--- Generating {n_per_weight} sequences with wt_weight={wt_w} ---")
        
        seqs = model.sample_unguided(
            structure_data=structure,
            wt_sequence=config["wt_sequence"],
            designable_positions=designable_positions,
            fixed_positions=fixed_positions,
            n_samples=n_per_weight,
            temperature=config["temperature"],
            wt_weight=wt_w,
        )
        
        for i, seq in enumerate(seqs):
            n_mut = mutation_count(seq, config["wt_sequence"])
            all_sequences.append(seq)
            all_metadata.append({
                "name": f"round1_wt{wt_w:.1f}_{i:04d}",
                "sequence": seq,
                "wt_weight": wt_w,
                "n_mutations": n_mut,
                "model": config["model"],
            })
    
    # Add wild-type copies as controls
    for i in range(3):
        all_sequences.append(config["wt_sequence"])
        all_metadata.append({
            "name": f"wildtype_control_{i}",
            "sequence": config["wt_sequence"],
            "wt_weight": -1,
            "n_mutations": 0,
            "model": "wildtype",
        })
    
    # Save outputs
    output_dir = Path(config["output"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FASTA
    names = [m["name"] for m in all_metadata]
    sequences_to_fasta(all_sequences, config["output"], names=names)
    logger.info(f"FASTA saved: {config['output']} ({len(all_sequences)} sequences)")
    
    # CSV
    df = pd.DataFrame(all_metadata)
    df.to_csv(config["output_csv"], index=False)
    logger.info(f"CSV saved: {config['output_csv']}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info(f"  Total sequences: {len(all_sequences)}")
    logger.info(f"  Mutation counts: {df['n_mutations'].describe().to_string()}")
    if len(all_sequences) > 1:
        diversity = compute_diversity(all_sequences[:100])  # sample for speed
        logger.info(f"  Diversity (sample): {diversity:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
