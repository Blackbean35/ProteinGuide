"""
ESM2 + MLP property predictor wrapped as a ProteinGuide BasePredictor.

Architecture
------------
  seq (str) → ESM2 backbone (frozen) → mean-pool over positions → 480-dim
            → Linear(480→hidden_size) → ReLU → Linear(hidden_size→1) → score

Key design decisions
--------------------
1. **Mean pooling** (not flattening): ESM2-35M produces 167×480=80K dims when
   flattened. Mean pooling → 480 dims, preventing overfitting and enabling
   a manageable head size.

2. **Embedding cache**: Since the backbone is frozen, embeddings never change.
   We compute them ONCE for the entire dataset before training and cache
   in memory as fp32 numpy arrays. This reduces per-epoch time from ~6 min
   to ~5 seconds.

3. **Rank-based label normalization**: TadABench-1M Value has a heavy long-tail
   distribution (most variants ~0, a few up to 14). Simple log+min-max pushes
   mean label to ~0.05, creating a degenerate MSE minimum at "predict always 0".
   Rank normalization (percentile, uniform [0,1]) avoids this.

4. **Masked positions** filled with WT before ESM2 tokenization — ESM2 cannot
   handle the ProteinGuide mask token (index 20).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional

from .base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)

# TadA8e wild-type sequence (167 AA)
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
    ESM2 (frozen) + MLP head predictor implementing ProteinGuide BasePredictor.

    Usage:
        # Train from scratch on TadABench-1M:
        predictor = ESM2MLPPredictor()
        predictor.train_on_tadabench("TadABench-1M/data")
        predictor.save("models/tada_esm2_predictor.pt")

        # Load and use for guidance:
        predictor = ESM2MLPPredictor.from_checkpoint("models/tada_esm2_predictor.pt")
        score = predictor.predict(encoded_seq)  # np.ndarray shape (167,)
    """

    def __init__(
        self,
        embed_name: str = "facebook/esm2_t12_35M_UR50D",
        hidden_size: int = 512,
        device: str = "auto",
        wt_sequence: str = TADA8E_WT,
    ):
        """
        Args:
            embed_name: HuggingFace ESM2 model ID.
            hidden_size: MLP hidden layer size. 512 is sufficient with mean pooling.
            device: 'cuda', 'cpu', or 'auto'.
            wt_sequence: WT sequence for filling masked positions during guidance.
        """
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.embed_name = embed_name
        self.seq_length = len(wt_sequence)
        self.wt_sequence = wt_sequence
        self._wt_encoded = np.array(
            [AA_TO_IDX.get(aa, 0) for aa in wt_sequence], dtype=np.int64
        )

        logger.info(f"Loading ESM2 backbone: {embed_name} → device={self._device}")
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(embed_name)
        self.backbone = AutoModel.from_pretrained(embed_name).to(self._device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # embed_dim: 480 (ESM2-35M), 640 (ESM2-150M), 1280 (ESM2-650M)
        self._embed_dim = self.backbone.config.hidden_size

        # MLP head: mean-pooled embed_dim → hidden_size → 1
        self.head = nn.Sequential(
            nn.Linear(self._embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        ).to(self._device)

        self.hidden_size = hidden_size
        logger.info(
            f"ESM2MLPPredictor | embed_dim={self._embed_dim} | "
            f"head: {self._embed_dim}→{hidden_size}→1 | device={self._device}"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fill_masked(self, encoded: np.ndarray) -> np.ndarray:
        """Replace ProteinGuide mask tokens (index 20) with WT amino acids."""
        filled = encoded.copy()
        mask_pos = filled == MASK_IDX
        filled[mask_pos] = self._wt_encoded[mask_pos]
        return filled

    def _encoded_to_str(self, encoded: np.ndarray) -> str:
        filled = self._fill_masked(encoded)
        return "".join(IDX_TO_AA.get(int(i), "A") for i in filled)

    @torch.no_grad()
    def _embed_mean(self, sequences: List[str]) -> torch.Tensor:
        """
        Embed sequences with ESM2, then mean-pool over sequence positions.

        Returns:
            Tensor shape (N, embed_dim), fp32 on CPU.
        """
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.seq_length + 2,  # +2 for [CLS] and [EOS]
        ).to(self._device)

        if self._device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.backbone(**inputs)
        else:
            out = self.backbone(**inputs)

        # Remove [CLS] and [EOS]; shape: (N, L, D)
        hidden = out.last_hidden_state[:, 1:-1, :]
        # Mean pool over sequence positions → (N, D)
        mean_emb = hidden.mean(dim=1)
        return mean_emb.to(torch.float32).cpu()

    def _precompute_embeddings(
        self,
        sequences,
        batch_size: int = 64,
        desc: str = "Embedding",
    ) -> np.ndarray:
        """
        Precompute mean-pooled ESM2 embeddings for all sequences.

        Returns:
            np.ndarray shape (N, embed_dim), fp32.
        """
        from tqdm import tqdm
        n = len(sequences)
        embeddings = np.zeros((n, self._embed_dim), dtype=np.float32)
        for start in tqdm(range(0, n, batch_size), desc=desc, unit="batch"):
            batch = [sequences[i] for i in range(start, min(start + batch_size, n))]
            emb = self._embed_mean(batch)  # (B, D) on CPU
            embeddings[start : start + len(batch)] = emb.numpy()
        return embeddings

    # ------------------------------------------------------------------ #
    # BasePredictor interface                                              #
    # ------------------------------------------------------------------ #

    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict activity for a single (possibly masked) sequence.

        Args:
            sequence: np.ndarray of shape (L,). Masked positions filled with WT.

        Returns:
            float: activity score in [0, 1].
        """
        seq_str = self._encoded_to_str(sequence)
        emb = self._embed_mean([seq_str]).to(self._device)  # (1, D)
        with torch.no_grad():
            logit = self.head(emb)
            score = torch.sigmoid(logit).item()
        return score

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict for a batch of encoded sequences.

        Args:
            sequences: np.ndarray of shape (N, L).

        Returns:
            np.ndarray of shape (N,).
        """
        seq_strs = [self._encoded_to_str(s) for s in sequences]
        emb = self._embed_mean(seq_strs).to(self._device)  # (N, D)
        with torch.no_grad():
            scores = torch.sigmoid(self.head(emb)).cpu().numpy().flatten()
        return scores

    def predict_str(self, sequence: str) -> float:
        """Predict activity for a plain AA string."""
        encoded = np.array([AA_TO_IDX.get(aa, 0) for aa in sequence], dtype=np.int64)
        return self.predict(encoded)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train_on_tadabench(
        self,
        data_dir: str = "TadABench-1M/data",
        n_epochs: int = 30,
        batch_size: int = 256,
        embed_batch_size: int = 64,
        lr: float = 3e-4,
        max_train_samples: Optional[int] = None,
    ):
        """
        Train MLP head on TadABench-1M AA split with embedding caching.

        Strategy
        --------
        1. Pre-compute ESM2 embeddings for all sequences ONCE (backbone is frozen).
        2. Train MLP head on cached embeddings for n_epochs (fast: ~5s/epoch).
        3. Use rank-based (percentile) label normalization → uniform [0,1] labels,
           avoiding the degenerate "predict-zero" MSE minimum.

        Args:
            data_dir: Path to TadABench-1M/data directory.
            n_epochs: Training epochs for the MLP head.
            batch_size: MLP training batch size (can be large since no backbone fwd).
            embed_batch_size: ESM2 batch size during embedding precomputation.
            lr: Learning rate for the MLP head (AdamW).
            max_train_samples: Cap training samples (None = use all 256K).
        """
        from datasets import load_from_disk

        # ---- Load data ------------------------------------------------
        logger.info("Loading TadABench-1M AA splits from disk...")
        train_ds = load_from_disk(str(Path(data_dir) / "all.AA.train"))
        val_ds   = load_from_disk(str(Path(data_dir) / "all.AA.val"))

        train_seqs = list(train_ds["Sequence"])
        train_vals = np.array(train_ds["Value"], dtype=np.float32)
        val_seqs   = list(val_ds["Sequence"])
        val_vals   = np.array(val_ds["Value"], dtype=np.float32)

        if max_train_samples:
            train_seqs = train_seqs[:max_train_samples]
            train_vals = train_vals[:max_train_samples]

        n_train = len(train_seqs)
        n_val   = len(val_seqs)
        logger.info(f"  Train: {n_train:,} | Val: {n_val:,}")
        logger.info(
            f"  Value stats | train: min={train_vals.min():.3f} "
            f"mean={train_vals.mean():.3f} max={train_vals.max():.3f}"
        )

        # ---- Label normalization (rank-based → uniform [0,1]) ----------
        # Rank normalization prevents the degenerate "predict all-zero" minimum.
        # mean=0.5 after normalization → MSE at all-zero = 0.25 (obviously bad)
        from scipy.stats import rankdata
        train_labels = (rankdata(train_vals) / n_train).astype(np.float32)
        val_labels   = (rankdata(val_vals) / n_val).astype(np.float32)
        logger.info(
            f"  Labels after rank-norm | train mean={train_labels.mean():.3f} "
            f"val mean={val_labels.mean():.3f}"
        )

        # ---- Pre-compute embeddings (ONCE, then cache) -----------------
        logger.info(
            f"Pre-computing ESM2 embeddings for {n_train:,} training sequences "
            f"(batch_size={embed_batch_size})..."
        )
        train_emb = self._precompute_embeddings(
            train_seqs, batch_size=embed_batch_size, desc="Embedding train"
        )  # (N_train, D)

        logger.info(f"Pre-computing embeddings for {n_val:,} val sequences...")
        val_emb = self._precompute_embeddings(
            val_seqs, batch_size=embed_batch_size, desc="Embedding val"
        )  # (N_val, D)

        # Convert to tensors (stays on CPU until batch is sent to device)
        train_emb_t  = torch.from_numpy(train_emb)
        train_label_t = torch.from_numpy(train_labels)
        val_emb_t    = torch.from_numpy(val_emb)

        logger.info(
            f"Embeddings cached | shape={train_emb.shape} | "
            f"memory={train_emb.nbytes / 1e6:.0f} MB"
        )

        # ---- Train MLP head on cached embeddings -----------------------
        optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
        criterion = nn.MSELoss()

        logger.info(
            f"Training MLP head for {n_epochs} epochs "
            f"(lr={lr}, batch={batch_size})..."
        )
        self.head.train()

        for epoch in range(1, n_epochs + 1):
            perm = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                # Fetch precomputed embeddings → move to device
                emb_b = train_emb_t[idx].to(self._device)
                lbl_b = train_label_t[idx].to(self._device)

                logits = self.head(emb_b).squeeze(-1)      # (B,)
                preds  = torch.sigmoid(logits)              # (B,) in [0,1]
                loss   = criterion(preds, lbl_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Spearman on val set using cached embeddings
            val_sp = self._eval_spearman_cached(val_emb_t, val_vals)
            logger.info(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"loss={avg_loss:.4f} | val_sp={val_sp:.4f}"
            )

        self.head.eval()
        logger.info("Training complete.")

    def _eval_spearman_cached(
        self,
        emb_tensor: torch.Tensor,
        true_values: np.ndarray,
        batch_size: int = 512,
    ) -> float:
        """Evaluate Spearman correlation using pre-cached embeddings."""
        from scipy.stats import spearmanr
        self.head.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(emb_tensor), batch_size):
                emb_b = emb_tensor[start : start + batch_size].to(self._device)
                logits = self.head(emb_b).squeeze(-1)
                scores = torch.sigmoid(logits).cpu().numpy()
                preds.extend(scores.tolist())
        self.head.train()
        sp, _ = spearmanr(true_values[: len(preds)], preds)
        return float(sp) if not np.isnan(sp) else 0.0

    # ------------------------------------------------------------------ #
    # Save / Load                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Save head weights + config."""
        torch.save(
            {
                "head_state": self.head.state_dict(),
                "embed_name": self.embed_name,
                "hidden_size": self.hidden_size,
                "wt_sequence": self.wt_sequence,
            },
            path,
        )
        logger.info(f"ESM2MLPPredictor saved → {path}")

    def load(self, path: str):
        """Load head weights from checkpoint."""
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self.head.load_state_dict(ckpt["head_state"])
        self.head.eval()
        logger.info(f"ESM2MLPPredictor loaded ← {path}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto",
    ) -> "ESM2MLPPredictor":
        """
        Load a fully initialised predictor from a saved .pt checkpoint.

        Args:
            checkpoint_path: Path to .pt file saved by .save().
            device: 'cuda', 'cpu', or 'auto'.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        predictor = cls(
            embed_name=ckpt["embed_name"],
            hidden_size=ckpt["hidden_size"],
            device=device,
            wt_sequence=ckpt.get("wt_sequence", TADA8E_WT),
        )
        predictor.head.load_state_dict(ckpt["head_state"])
        predictor.head.eval()
        logger.info(f"Loaded ESM2MLPPredictor ← {checkpoint_path}")
        return predictor
