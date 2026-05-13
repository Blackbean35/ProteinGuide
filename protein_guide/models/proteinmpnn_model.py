"""
ProteinMPNN wrapper for ProteinGuide framework.

ProteinMPNN is an any-order autoregressive (AO-ARM) inverse folding model.
It can also be sampled as a masked flow matching model via training equivalence.

Requires: git clone https://github.com/dauparas/ProteinMPNN.git
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, List, Optional
from .base_model import BaseGenerativeModel, _softmax
from ..data.sequence_utils import (
    AA_ALPHABET, AA_TO_IDX, IDX_TO_AA, MASK_IDX, VOCAB_SIZE,
    decode_sequence, create_masked_sequence,
)
import logging

logger = logging.getLogger(__name__)
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
MPNN_AA_TO_IDX = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}


class ProteinMPNNModel(BaseGenerativeModel):
    """ProteinMPNN wrapper for inverse folding and guided generation."""

    def __init__(self, proteinmpnn_dir="./ProteinMPNN",
                 model_name="v_48_020", device_str="cuda", n_chains=1):
        self._device = torch.device(device_str)
        self.proteinmpnn_dir = Path(proteinmpnn_dir)
        self.model_name = model_name
        self.n_chains = n_chains
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        mpnn_dir = self.proteinmpnn_dir
        if not mpnn_dir.exists():
            raise FileNotFoundError(
                f"ProteinMPNN not found at {mpnn_dir}. "
                f"Run: git clone https://github.com/dauparas/ProteinMPNN.git"
            )
        sys.path.insert(0, str(mpnn_dir))
        from protein_mpnn_utils import ProteinMPNN as MPNNClass
        ckpt_path = mpnn_dir / "vanilla_model_weights" / f"{self.model_name}.pt"
        if not ckpt_path.exists():
            ckpt_path = mpnn_dir / "model_weights" / f"{self.model_name}.pt"
        checkpoint = torch.load(ckpt_path, map_location=self._device)
        model = MPNNClass(
            num_letters=21, node_features=128, edge_features=128,
            hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
            augment_eps=0.0, k_neighbors=checkpoint.get("num_edges", 48),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device).eval()
        self._model = model
        logger.info("ProteinMPNN loaded successfully")

    def forward(self, sequence, structure_data, temperature=1.0):
        self._ensure_loaded()
        L = len(sequence)
        coords = np.stack([structure_data["N"], structure_data["CA"],
                          structure_data["C"], structure_data["O"]], axis=1)
        if self.n_chains > 1:
            coords = np.tile(coords, (self.n_chains, 1, 1))
        X = torch.tensor(coords, dtype=torch.float32, device=self._device).unsqueeze(0)
        # Encode
        S = self._seq_to_mpnn(sequence)
        if self.n_chains > 1:
            S = np.tile(S, self.n_chains)
        S_t = torch.tensor(S, dtype=torch.long, device=self._device).unsqueeze(0)
        mask = torch.ones(1, X.shape[1], dtype=torch.float32, device=self._device)
        chain_enc = torch.zeros(1, X.shape[1], dtype=torch.long, device=self._device)
        residue_idx = torch.arange(X.shape[1], device=self._device).unsqueeze(0)
        with torch.no_grad():
            log_probs = self._model(X, S_t, mask, chain_enc, residue_idx)
        lp = log_probs[0].cpu().numpy()
        logits_20 = np.zeros((lp.shape[0], VOCAB_SIZE), dtype=np.float32)
        for our_idx, aa in IDX_TO_AA.items():
            if aa in MPNN_AA_TO_IDX:
                logits_20[:, our_idx] = lp[:, MPNN_AA_TO_IDX[aa]]
        if self.n_chains > 1:
            logits_20 = logits_20.reshape(self.n_chains, L, VOCAB_SIZE).mean(0)
        else:
            logits_20 = logits_20[:L]
        return logits_20

    def sample_unguided(self, structure_data, wt_sequence, designable_positions,
                        fixed_positions=None, n_samples=1, temperature=0.5,
                        wt_weight=0.0, **kwargs):
        self._ensure_loaded()
        actual = set(designable_positions)
        if fixed_positions:
            actual -= set(fixed_positions)
        actual = sorted(actual)
        generated = []
        for si in range(n_samples):
            seq = create_masked_sequence(wt_sequence, designable_positions, fixed_positions)
            order = np.array(actual)
            np.random.shuffle(order)
            for pos in order:
                logits = self.forward(seq, structure_data)
                if wt_weight > 0:
                    logits = self.apply_wt_weight(logits, wt_sequence, wt_weight, actual)
                probs = _softmax(logits[pos] / max(temperature, 1e-8))
                seq[pos] = np.random.choice(VOCAB_SIZE, p=probs)
            generated.append(decode_sequence(seq))
            if (si+1) % 10 == 0:
                logger.info(f"  Generated {si+1}/{n_samples}")
        return generated

    def _seq_to_mpnn(self, sequence):
        S = np.zeros(len(sequence), dtype=np.int64)
        for i, idx in enumerate(sequence):
            if idx == MASK_IDX:
                S[i] = 20
            elif idx < VOCAB_SIZE:
                aa = IDX_TO_AA[idx]
                S[i] = MPNN_AA_TO_IDX.get(aa, 20)
        return S

    @property
    def device(self):
        return self._device
