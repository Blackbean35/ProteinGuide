"""
Masking utilities for noisy predictor training and guided sampling.
Implements the masking noise process used in masked flow matching / MLMs.
"""

import numpy as np
import torch
from typing import Optional


def random_mask(
    sequence: np.ndarray,
    mask_rate: float,
    designable_positions: Optional[np.ndarray] = None,
    mask_idx: int = 20,
) -> np.ndarray:
    """
    Randomly mask positions in a sequence at a given rate.

    Args:
        sequence: Integer-encoded sequence, shape (L,).
        mask_rate: Fraction of (designable) positions to mask, in [0, 1].
        designable_positions: If given, only mask these positions. Otherwise mask all.
        mask_idx: Index used for masked tokens.

    Returns:
        Masked copy of the sequence.
    """
    masked = sequence.copy()

    if designable_positions is not None:
        positions = designable_positions
    else:
        positions = np.arange(len(sequence))

    n_to_mask = max(1, int(len(positions) * mask_rate))
    mask_positions = np.random.choice(positions, size=n_to_mask, replace=False)
    masked[mask_positions] = mask_idx

    return masked


def sample_mask_rate(schedule: str = "uniform") -> float:
    """
    Sample a masking rate from the noise schedule distribution.

    Args:
        schedule: Type of schedule.
            - 'uniform': t ~ U(0, 1)
            - 'cosine': cosine schedule

    Returns:
        Float masking rate in (0, 1].
    """
    if schedule == "uniform":
        return np.random.uniform(0.01, 1.0)
    elif schedule == "cosine":
        u = np.random.uniform(0, 1)
        return 1.0 - np.cos(u * np.pi / 2)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def create_noisy_dataset(
    sequences: np.ndarray,
    labels: np.ndarray,
    n_noisy_copies: int = 10,
    designable_positions: Optional[np.ndarray] = None,
    schedule: str = "uniform",
    mask_idx: int = 20,
) -> tuple:
    """
    Create a noisy dataset by generating multiple masked copies of each sequence.
    Each copy uses a different random masking rate sampled from the noise schedule.

    This is used to train the "noisy predictor" needed for guidance.

    Args:
        sequences: Integer-encoded sequences, shape (N, L).
        labels: Labels, shape (N,) or (N, C).
        n_noisy_copies: Number of noisy copies per sequence.
        designable_positions: Positions eligible for masking.
        schedule: Noise schedule for masking rate.
        mask_idx: Mask token index.

    Returns:
        Tuple of (noisy_sequences, noisy_labels, mask_rates):
            - noisy_sequences: shape (N * n_noisy_copies, L)
            - noisy_labels: shape (N * n_noisy_copies,) or (N * n_noisy_copies, C)
            - mask_rates: shape (N * n_noisy_copies,)
    """
    N, L = sequences.shape
    all_noisy = []
    all_labels = []
    all_rates = []

    for i in range(N):
        for _ in range(n_noisy_copies):
            rate = sample_mask_rate(schedule)
            noisy_seq = random_mask(
                sequences[i], rate, designable_positions, mask_idx
            )
            all_noisy.append(noisy_seq)
            all_labels.append(labels[i])
            all_rates.append(rate)

    return (
        np.array(all_noisy),
        np.array(all_labels),
        np.array(all_rates),
    )


def mask_sequence_torch(
    sequence: torch.Tensor,
    mask_rate: float,
    designable_mask: Optional[torch.Tensor] = None,
    mask_idx: int = 20,
) -> torch.Tensor:
    """
    Randomly mask a sequence tensor.

    Args:
        sequence: shape (L,) integer tensor.
        mask_rate: Fraction of positions to mask.
        designable_mask: Boolean tensor of shape (L,) — True = designable.
        mask_idx: Mask token index.

    Returns:
        Masked sequence tensor.
    """
    masked = sequence.clone()
    if designable_mask is not None:
        positions = torch.where(designable_mask)[0]
    else:
        positions = torch.arange(len(sequence))

    n_to_mask = max(1, int(len(positions) * mask_rate))
    perm = torch.randperm(len(positions))[:n_to_mask]
    mask_positions = positions[perm]
    masked[mask_positions] = mask_idx

    return masked
