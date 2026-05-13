"""
Abstract base class for generative models used in ProteinGuide.
All generative models (ESM3, ProteinMPNN) must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class BaseGenerativeModel(ABC):
    """
    Interface for protein sequence generative models.

    All models must be able to:
    1. Compute logits for masked positions given structure context
    2. Provide rate matrices for flow-matching sampling
    3. Generate unguided samples
    """

    @abstractmethod
    def forward(
        self,
        sequence: np.ndarray,
        structure_data: Dict[str, np.ndarray],
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Forward pass: compute logits for each position.

        Args:
            sequence: Integer-encoded sequence, shape (L,). Masked positions = 20.
            structure_data: Dictionary from load_pdb_structure().
            temperature: Sampling temperature.

        Returns:
            Logits array of shape (L, 20) — probabilities over 20 amino acids
            for each position. Non-designable positions may be ignored.
        """
        pass

    @abstractmethod
    def sample_unguided(
        self,
        structure_data: Dict[str, np.ndarray],
        wt_sequence: str,
        designable_positions: List[int],
        fixed_positions: Optional[List[int]] = None,
        n_samples: int = 1,
        temperature: float = 0.5,
        wt_weight: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate unguided sequences by sampling from the model.

        Args:
            structure_data: Structure dictionary.
            wt_sequence: Wild-type sequence.
            designable_positions: 0-indexed positions to design.
            fixed_positions: Positions within designable region to keep fixed.
            n_samples: Number of sequences to generate.
            temperature: Sampling temperature.
            wt_weight: Bias toward wild-type amino acids (0 = no bias).

        Returns:
            List of generated sequence strings.
        """
        pass

    def get_logits_for_position(
        self,
        sequence: np.ndarray,
        position: int,
        structure_data: Dict[str, np.ndarray],
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Get probability distribution over amino acids for a single position.
        Default implementation calls forward() and extracts the position.

        Args:
            sequence: Current (partially masked) sequence.
            position: Position index to get logits for.
            structure_data: Structure context.
            temperature: Sampling temperature.

        Returns:
            Probability vector of shape (20,).
        """
        all_logits = self.forward(sequence, structure_data, temperature)
        logits = all_logits[position]  # (20,)

        # Apply temperature and softmax
        logits = logits / max(temperature, 1e-8)
        probs = _softmax(logits)
        return probs

    def apply_wt_weight(
        self,
        logits: np.ndarray,
        wt_sequence: str,
        wt_weight: float,
        designable_positions: List[int],
    ) -> np.ndarray:
        """
        Apply wild-type weight bias to logits.
        Increases the logit of the wild-type amino acid at each designable position.

        logits_biased[pos, wt_aa] = logits[pos, wt_aa] + wt_weight

        Args:
            logits: shape (L, 20)
            wt_sequence: wild-type sequence string.
            wt_weight: bias magnitude (0 = no bias).
            designable_positions: positions to apply bias.

        Returns:
            Modified logits of shape (L, 20).
        """
        from ..data.sequence_utils import AA_TO_IDX

        biased = logits.copy()
        for pos in designable_positions:
            if pos < len(wt_sequence):
                wt_aa = wt_sequence[pos]
                if wt_aa in AA_TO_IDX:
                    biased[pos, AA_TO_IDX[wt_aa]] += wt_weight
        return biased

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device this model is on."""
        pass


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
