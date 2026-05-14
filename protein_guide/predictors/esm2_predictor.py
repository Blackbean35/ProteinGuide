"""
ESM2 + MLP property predictor wrapped as a ProteinGuide BasePredictor.

This bridges the TadABench-1M training pipeline (ESM2 backbone + MLP head,
trained on ~256K TadA variant sequences) with the ProteinGuide guidance
algorithms (DEG / TAG).

The predictor takes a numpy integer-encoded sequence (shape: (L,)) and returns
a scalar activity score. For guidance, it is evaluated at every decoding step
on partially-decoded (masked) sequences — masked positions are replaced with
the wild-type amino acid before embedding, because ESM2 cannot handle the
ProteinGuide mask token (index 20).
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional

from .base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)

# WT sequence used as fallback for masked positions during guidance
TADA8E_WT = (
    "MSEVEFSHEYWMRHALTLAKRARDEREVPVGAVLVLNNRVIGEGWNRAIGLHDPTAHAEIMALRQGG"
    "LVMQNYRLIDATLYVTFEPCVMCAGAMIHSRIGRVVFGVRNAKTGAAGSLMDVLHYPGMNHRVEITE"
    "GILADECAALLCDFYRMPRQVFNAQKKAQSSTD"
)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_ALPHABET)}
MASK_IDX = 20  # ProteinGuide mask token index


class ESM2MLPPredictor(BasePredictor):
    """
    ESM2 + MLP head predictor implementing the ProteinGuide BasePredictor interface.

    The model architecture mirrors TadABench-1M's training setup:
      - Backbone: ESM2 (frozen, bf16 on GPU)
      - Head: Linear MLP (fp32)

    Masked positions (index 20 in ProteinGuide encoding) are replaced with
    the wild-type amino acid before ESM2 tokenization, ensuring the backbone
    sees a valid protein sequence at every decoding step.

    Usage:
        predictor = ESM2MLPPredictor.from_checkpoint("models/tada_esm2_predictor.pt")
        score = predictor.predict(encoded_seq)   # shape (167,) numpy int array
    """

    def __init__(
        self,
        embed_name: str = "facebook/esm2_t12_35M_UR50D",
        hidden_size: int = 1024,
        seq_length: int = 167,
        device: str = "auto",
        wt_sequence: str = TADA8E_WT,
    ):
        """
        Args:
            embed_name: HuggingFace model name for ESM2 backbone.
            hidden_size: Hidden layer size in MLP head.
            seq_length: Protein sequence length (167 for TadA).
            device: 'cuda', 'cpu', or 'auto' (auto-detect).
            wt_sequence: Wild-type sequence used to fill masked positions.
        """
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.embed_name = embed_name
        self.seq_length = seq_length
        self.wt_sequence = wt_sequence
        self._wt_encoded = np.array(
            [AA_TO_IDX.get(aa, 0) for aa in wt_sequence], dtype=np.int64
        )

        logger.info(f"Loading ESM2 backbone: {embed_name}")
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(embed_name)
        self.backbone = AutoModel.from_pretrained(embed_name).to(self._device)
        self.backbone.eval()
        # Freeze backbone — only MLP head is trained
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        embed_dim = self.backbone.config.hidden_size  # 480 for ESM2-35M
        self.head = nn.Sequential(
            nn.Linear(seq_length * embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(self._device)

        self._embed_dim = embed_dim
        logger.info(
            f"ESM2MLPPredictor ready | backbone={embed_name} | "
            f"embed_dim={embed_dim} | device={self._device}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_masked(self, encoded: np.ndarray) -> np.ndarray:
        """Replace mask tokens (index 20) with wild-type amino acid."""
        filled = encoded.copy()
        mask_positions = filled == MASK_IDX
        filled[mask_positions] = self._wt_encoded[mask_positions]
        return filled

    def _encoded_to_str(self, encoded: np.ndarray) -> str:
        """Convert integer-encoded sequence to string (filling masks with WT)."""
        filled = self._fill_masked(encoded)
        return "".join(IDX_TO_AA.get(int(i), "A") for i in filled)

    def _embed(self, sequences: List[str]) -> torch.Tensor:
        """
        Embed protein sequences with ESM2.

        Returns:
            Tensor of shape (N, seq_length * embed_dim), fp32.
        """
        inputs = self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)

        with torch.no_grad():
            if self._device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self.backbone(**inputs)
            else:
                out = self.backbone(**inputs)

        # Remove [CLS] and [EOS] special tokens; shape: (N, L, D)
        emb = out.last_hidden_state[:, 1:-1, :]
        # Flatten to (N, L*D)
        return emb.to(torch.float32).view(len(sequences), -1)

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict activity score for a single (possibly masked) sequence.

        Args:
            sequence: np.ndarray of shape (L,) with integer indices.
                      Masked positions (index 20) are filled with WT.

        Returns:
            float: predicted activity score (sigmoid of logit, 0–1 scale).
        """
        seq_str = self._encoded_to_str(sequence)
        emb = self._embed([seq_str])  # (1, L*D)

        with torch.no_grad():
            logit = self.head(emb)
            score = torch.sigmoid(logit).item()
        return score

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict for a batch of sequences.

        Args:
            sequences: np.ndarray of shape (N, L).

        Returns:
            np.ndarray of shape (N,) with activity scores.
        """
        seq_strs = [self._encoded_to_str(s) for s in sequences]
        emb = self._embed(seq_strs)  # (N, L*D)

        with torch.no_grad():
            logits = self.head(emb)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
        return scores

    def predict_str(self, sequence: str) -> float:
        """Predict activity for a plain amino acid string (no masking needed)."""
        encoded = np.array([AA_TO_IDX.get(aa, 0) for aa in sequence], dtype=np.int64)
        return self.predict(encoded)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_tadabench(
        self,
        data_dir: str = "TadABench-1M/data",
        n_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-4,
        threshold: Optional[float] = None,
        max_train_samples: Optional[int] = None,
    ):
        """
        Train the MLP head on TadABench-1M (AA split).

        The backbone (ESM2) is frozen throughout. Only the MLP head is trained.
        Labels are normalized to [0, 1] by sigmoid of log(Value), making the
        MSE loss work in a bounded regression setting.

        Args:
            data_dir: Path to TadABench-1M data directory.
            n_epochs: Number of training epochs.
            batch_size: Batch size (limit based on GPU memory).
            lr: Learning rate for the MLP head.
            threshold: If set, converts to binary classification (Value >= threshold).
            max_train_samples: Limit training samples (for quick tests).
        """
        from datasets import load_from_disk

        logger.info("Loading TadABench-1M AA train split...")
        train_ds = load_from_disk(str(Path(data_dir) / "all.AA.train"))
        val_ds = load_from_disk(str(Path(data_dir) / "all.AA.val"))

        train_seqs = train_ds["Sequence"]
        train_vals = np.array(train_ds["Value"], dtype=np.float32)
        val_seqs = val_ds["Sequence"]
        val_vals = np.array(val_ds["Value"], dtype=np.float32)

        if max_train_samples:
            train_seqs = train_seqs[:max_train_samples]
            train_vals = train_vals[:max_train_samples]

        logger.info(f"  Train: {len(train_seqs)} | Val: {len(val_seqs)}")

        # Normalize labels: log-transform then min-max to [0,1]
        train_labels = self._normalize_labels(train_vals, threshold)
        val_labels   = self._normalize_labels(val_vals,   threshold)

        optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.MSELoss() if threshold is None else nn.BCEWithLogitsLoss()

        self.head.train()
        n_train = len(train_seqs)

        for epoch in range(1, n_epochs + 1):
            # Shuffle
            perm = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                batch_seqs = [train_seqs[i] for i in idx]
                batch_labels = torch.tensor(
                    train_labels[idx], dtype=torch.float32, device=self._device
                )

                emb = self._embed(batch_seqs)
                logits = self.head(emb).squeeze()

                if threshold is None:
                    # Regression: predict sigmoid(logit) ≈ label
                    preds = torch.sigmoid(logits)
                    loss = criterion(preds, batch_labels)
                else:
                    loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation every epoch
            val_sp = self._eval_spearman(val_seqs[:500], val_vals[:500])
            logger.info(
                f"Epoch {epoch:3d}/{n_epochs} | loss={avg_loss:.4f} | val_sp={val_sp:.3f}"
            )

        self.head.eval()
        logger.info("Training complete.")

    def _normalize_labels(self, values: np.ndarray, threshold) -> np.ndarray:
        """Normalize raw TadABench-1M Value to [0, 1]."""
        if threshold is not None:
            return (values >= threshold).astype(np.float32)
        # Log-scale then min-max normalize
        log_vals = np.log1p(values)
        mn, mx = log_vals.min(), log_vals.max()
        if mx > mn:
            return ((log_vals - mn) / (mx - mn)).astype(np.float32)
        return np.zeros_like(values)

    def _eval_spearman(self, sequences, true_values: np.ndarray) -> float:
        """Compute Spearman correlation on a subset."""
        from scipy.stats import spearmanr
        preds = []
        self.head.eval()
        with torch.no_grad():
            for i in range(0, len(sequences), 64):
                batch = sequences[i : i + 64]
                emb = self._embed(batch)
                logits = self.head(emb).squeeze()
                scores = torch.sigmoid(logits).cpu().numpy()
                if scores.ndim == 0:
                    scores = scores.reshape(1)
                preds.extend(scores.tolist())
        self.head.train()
        sp, _ = spearmanr(true_values[: len(preds)], preds)
        return float(sp) if not np.isnan(sp) else 0.0

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save MLP head weights + config."""
        torch.save(
            {
                "head_state": self.head.state_dict(),
                "embed_name": self.embed_name,
                "seq_length": self.seq_length,
                "hidden_size": self.head[0].out_features,
                "wt_sequence": self.wt_sequence,
            },
            path,
        )
        logger.info(f"ESM2MLPPredictor saved to {path}")

    def load(self, path: str):
        """Load MLP head weights from checkpoint."""
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self.head.load_state_dict(ckpt["head_state"])
        self.head.eval()
        logger.info(f"ESM2MLPPredictor loaded from {path}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto",
    ) -> "ESM2MLPPredictor":
        """
        Load a fully initialised predictor from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pt file saved by .save().
            device: 'cuda', 'cpu', or 'auto'.

        Returns:
            Loaded ESM2MLPPredictor instance.
        """
        ckpt = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        predictor = cls(
            embed_name=ckpt["embed_name"],
            hidden_size=ckpt["hidden_size"],
            seq_length=ckpt["seq_length"],
            device=device,
            wt_sequence=ckpt.get("wt_sequence", TADA8E_WT),
        )
        predictor.head.load_state_dict(ckpt["head_state"])
        predictor.head.eval()
        logger.info(f"Loaded ESM2MLPPredictor from {checkpoint_path}")
        return predictor
