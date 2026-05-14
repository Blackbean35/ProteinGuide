#!/usr/bin/env python3
"""
Step 0: Train TadA Activity Predictor from TadABench-1M

This is a self-contained script that:
  1. Loads TadABench-1M AA split (train / val / test)
  2. Trains an ESM2 + MLP head predictor (backbone frozen, only head trained)
  3. Evaluates on val and test sets (Spearman, Recall@10%, NDCG@10%)
  4. Saves the predictor to models/tada_esm2_predictor.pt

The saved predictor can be directly loaded by scripts/03_guided_generation.py
using --predictor_type esm2.

Usage:
    # Quick training (ESM2-35M, ~30 min on GPU):
    python scripts/00_train_tada_predictor.py

    # Custom backbone or settings:
    python scripts/00_train_tada_predictor.py \\
        --backbone facebook/esm2_t30_150M_UR50D \\
        --hidden_size 2048 \\
        --n_epochs 30 \\
        --batch_size 32 \\
        --output models/tada_esm2_150M_predictor.pt

    # Quick smoke test (small data):
    python scripts/00_train_tada_predictor.py --max_train 2000 --n_epochs 3
"""

import argparse
import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_guide.predictors.esm2_predictor import ESM2MLPPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# TadA8e wild-type sequence (167 AA)
TADA8E_WT = (
    "MSEVEFSHEYWMRHALTLAKRARDEREVPVGAVLVLNNRVIGEGWNRAIGLHDPTAHAEIMALRQGG"
    "LVMQNYRLIDATLYVTFEPCVMCAGAMIHSRIGRVVFGVRNAKTGAAGSLMDVLHYPGMNHRVEITE"
    "GILADECAALLCDFYRMPRQVFNAQKKAQSSTD"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ESM2+MLP predictor on TadABench-1M"
    )
    parser.add_argument(
        "--data_dir",
        default="TadABench-1M/data",
        help="Path to TadABench-1M data directory (contains all.AA.train/ etc.)",
    )
    parser.add_argument(
        "--backbone",
        default="facebook/esm2_t12_35M_UR50D",
        choices=[
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
            "facebook/esm2_t33_650M_UR50D",
        ],
        help="ESM2 backbone model (larger = better but slower)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="MLP hidden layer size",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (reduce if OOM on GPU)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for MLP head",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Limit training samples (for quick tests; None = use all 256K)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'cuda', 'cpu', or 'auto' (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        default="models/tada_esm2_predictor.pt",
        help="Output path for trained predictor",
    )
    return parser.parse_args()


def evaluate_full(predictor, data_dir: str, split: str = "test"):
    """Evaluate Spearman, Recall@10%, NDCG@10% on val or test set."""
    from datasets import load_from_disk
    from scipy.stats import spearmanr

    logger.info(f"Evaluating on {split} set...")
    ds = load_from_disk(str(Path(data_dir) / f"all.AA.{split}"))
    seqs = ds["Sequence"]
    vals = np.array(ds["Value"], dtype=np.float32)

    # Predict in batches
    preds = []
    batch_size = 64
    predictor.head.eval()
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start : start + batch_size]
        emb = predictor._embed(batch)
        import torch
        with torch.no_grad():
            scores = torch.sigmoid(predictor.head(emb)).squeeze()
        if scores.ndim == 0:
            scores = scores.reshape(1)
        preds.extend(scores.cpu().numpy().tolist())

    preds = np.array(preds)
    labels = vals[: len(preds)]

    # Spearman
    sp, _ = spearmanr(labels, preds)

    # Recall@10%
    n = len(labels)
    k = max(1, int(n * 0.1))
    top_thresh = np.percentile(labels, 90)
    positive_idx = np.where(labels >= top_thresh)[0]
    top_pred_idx = np.argsort(preds)[::-1][:k]
    recall10 = len(np.intersect1d(top_pred_idx, positive_idx)) / max(len(positive_idx), 1)

    # NDCG@10%
    gains = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
    order = np.argsort(preds)[::-1][:k]
    dcg = np.sum(gains[order] / np.log2(np.arange(2, k + 2)))
    ideal = np.argsort(gains)[::-1][:k]
    idcg = np.sum(gains[ideal] / np.log2(np.arange(2, k + 2)))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    logger.info(
        f"  [{split}] Spearman={sp:.4f} | Recall@10%={recall10:.4f} | NDCG@10%={ndcg10:.4f}"
    )
    return {"spearman": float(sp), "recall_at_10pct": recall10, "ndcg_at_10pct": ndcg10}


def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("ProteinGuide — Step 0: TadA Predictor Training")
    logger.info("=" * 65)
    logger.info(f"  Backbone   : {args.backbone}")
    logger.info(f"  Hidden size: {args.hidden_size}")
    logger.info(f"  Epochs     : {args.n_epochs}")
    logger.info(f"  Batch size : {args.batch_size}")
    logger.info(f"  Data dir   : {args.data_dir}")
    logger.info(f"  Output     : {args.output}")
    if args.max_train:
        logger.info(f"  ⚠ Max train samples: {args.max_train} (smoke-test mode)")
    logger.info("")

    # Build predictor
    predictor = ESM2MLPPredictor(
        embed_name=args.backbone,
        hidden_size=args.hidden_size,
        seq_length=167,
        device=args.device,
        wt_sequence=TADA8E_WT,
    )

    # Train on TadABench-1M
    predictor.train_on_tadabench(
        data_dir=args.data_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threshold=None,            # Regression mode (normalized [0,1])
        max_train_samples=args.max_train,
    )

    # Evaluate on val and test
    val_metrics  = evaluate_full(predictor, args.data_dir, split="val")
    test_metrics = evaluate_full(predictor, args.data_dir, split="test")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(str(output_path))

    logger.info("")
    logger.info("=" * 65)
    logger.info("Training complete!")
    logger.info(f"  Val  Spearman : {val_metrics['spearman']:.4f}")
    logger.info(f"  Test Spearman : {test_metrics['spearman']:.4f}")
    logger.info(f"  Test Recall@10%: {test_metrics['recall_at_10pct']:.4f}")
    logger.info(f"  Test NDCG@10%  : {test_metrics['ndcg_at_10pct']:.4f}")
    logger.info(f"  Saved to: {output_path}")
    logger.info("")
    logger.info("Next step — run guided generation:")
    logger.info(
        f"  python scripts/03_guided_generation.py \\\n"
        f"      --predictor {output_path} \\\n"
        f"      --predictor_type esm2 \\\n"
        f"      --config configs/tada_config.yaml \\\n"
        f"      --algorithm deg --gamma 100 --n_samples 500"
    )
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
