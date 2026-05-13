#!/usr/bin/env python3
"""
Round 1.5: Train Predictor from Experimental Data

Trains a property predictor on experimental assay data (e.g., log enrichment
from selection experiments). The predictor is trained in two phases:
1. Clean data training (standard supervision)
2. Noisy data training (masked input augmentation for guidance compatibility)

The trained predictor can then be loaded in Round 2 for guided generation.

Usage:
    python scripts/02_train_predictor.py \
        --data data/round1_results.csv \
        --wt_sequence MSEVEFSHE... \
        --designable_start 82 --designable_end 167 \
        --label_column log_enrichment \
        --threshold 0.0 \
        --output models/predictor.pt
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_guide.data.sequence_utils import encode_sequence, AA_ALPHABET
from protein_guide.predictors.linear_predictor import LinearPairwisePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train property predictor")
    
    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="CSV file with sequences and labels")
    parser.add_argument("--sequence_column", type=str, default="sequence",
                        help="Column name for sequences")
    parser.add_argument("--label_column", type=str, default="log_enrichment",
                        help="Column name for property values")
    
    # Label processing
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold for binary classification (value >= threshold → positive)")
    parser.add_argument("--filter_range", type=str, default=None,
                        help="Exclude sequences with labels in this range, e.g. '-0.25,0.25'")
    parser.add_argument("--replicate_column", type=str, default=None,
                        help="Column for replicate values (to filter high-variance sequences)")
    parser.add_argument("--max_replicate_diff", type=float, default=0.25,
                        help="Max allowed difference between replicates")
    
    # Sequence config
    parser.add_argument("--wt_sequence", type=str, required=True)
    parser.add_argument("--designable_start", type=int, required=True)
    parser.add_argument("--designable_end", type=int, required=True)
    
    # Training
    parser.add_argument("--predictor_type", type=str, default="linear_pairwise",
                        choices=["linear_pairwise"])
    parser.add_argument("--reg_lambda", type=float, default=10.0)
    parser.add_argument("--clean_epochs", type=int, default=400)
    parser.add_argument("--noisy_epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    
    # Validation
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--n_cv_splits", type=int, default=5,
                        help="Number of cross-validation splits for AUROC")
    
    # Output
    parser.add_argument("--output", type=str, default="models/predictor.pt")
    
    return parser.parse_args()


def load_and_preprocess_data(args):
    """Load CSV, filter, and create binary labels."""
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"  Loaded {len(df)} sequences")
    
    # Filter by replicate consistency
    if args.replicate_column and args.replicate_column in df.columns:
        before = len(df)
        df = df[df[args.replicate_column].abs() <= args.max_replicate_diff]
        logger.info(f"  Filtered by replicate consistency: {before} → {len(df)}")
    
    # Filter by label range (exclude ambiguous zone)
    if args.filter_range:
        low, high = [float(x) for x in args.filter_range.split(",")]
        before = len(df)
        df = df[(df[args.label_column] <= low) | (df[args.label_column] >= high)]
        logger.info(f"  Filtered range [{low}, {high}]: {before} → {len(df)}")
    
    # Create binary labels
    df["label"] = (df[args.label_column] >= args.threshold).astype(int)
    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos
    logger.info(f"  Positive: {n_pos}, Negative: {n_neg} (threshold={args.threshold})")
    
    return df


def encode_sequences(df, args):
    """Encode sequences to numpy arrays."""
    sequences = []
    for seq_str in df[args.sequence_column]:
        encoded = encode_sequence(seq_str)
        sequences.append(encoded)
    return np.array(sequences)


def evaluate_auroc(predictor, sequences, labels, n_splits=5):
    """Evaluate AUROC via stratified cross-validation."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aurocs = []
    
    for train_idx, val_idx in skf.split(sequences, labels):
        val_seqs = sequences[val_idx]
        val_labels = labels[val_idx]
        preds = predictor.predict_batch(val_seqs)
        try:
            auroc = roc_auc_score(val_labels, preds)
            aurocs.append(auroc)
        except ValueError:
            pass
    
    return np.mean(aurocs) if aurocs else 0.0


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("ProteinGuide — Predictor Training")
    logger.info("=" * 60)
    
    # Load data
    df = load_and_preprocess_data(args)
    
    # Encode sequences
    sequences = encode_sequences(df, args)
    labels = df["label"].values.astype(np.float32)
    L = sequences.shape[1]
    
    # Set up designable positions
    designable_positions = list(range(args.designable_start, args.designable_end))
    
    logger.info(f"Sequence length: {L}")
    logger.info(f"Designable positions: {len(designable_positions)}")
    
    # Create predictor
    predictor = LinearPairwisePredictor(
        seq_length=L,
        designable_positions=designable_positions,
        reg_lambda=args.reg_lambda,
        device=args.device,
    )
    
    # Phase 1: Clean training
    logger.info("\n--- Phase 1: Clean data training ---")
    predictor.train_clean(
        sequences=sequences,
        labels=labels,
        n_epochs=args.clean_epochs,
        lr=args.lr,
    )
    
    # Phase 2: Noisy training
    logger.info("\n--- Phase 2: Noisy data training ---")
    designable_mask = np.zeros(L, dtype=bool)
    designable_mask[args.designable_start:args.designable_end] = True
    
    predictor.train_noisy(
        sequences=sequences,
        labels=labels,
        designable_mask=designable_mask,
        n_epochs=args.noisy_epochs,
        lr=args.lr,
    )
    
    # Evaluate
    logger.info("\n--- Evaluation ---")
    preds = predictor.predict_batch(sequences)
    accuracy = ((preds > 0.5) == labels).mean()
    logger.info(f"Training accuracy: {accuracy:.3f}")
    logger.info(f"Mean predicted score (positive): {preds[labels==1].mean():.3f}")
    logger.info(f"Mean predicted score (negative): {preds[labels==0].mean():.3f}")
    
    try:
        auroc = evaluate_auroc(predictor, sequences, labels, args.n_cv_splits)
        logger.info(f"Cross-validated AUROC: {auroc:.3f}")
    except ImportError:
        logger.warning("sklearn not installed, skipping AUROC evaluation")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(str(output_path))
    
    logger.info(f"\nPredictor saved to {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
