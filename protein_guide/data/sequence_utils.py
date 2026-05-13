"""
Sequence encoding/decoding utilities for protein sequences.
Handles conversion between string sequences, integer indices, and one-hot encodings.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple
from pathlib import Path


# Standard 20 amino acid alphabet
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_ALPHABET)}
VOCAB_SIZE = len(AA_ALPHABET)  # 20

# Special tokens
MASK_TOKEN = "<mask>"
MASK_IDX = VOCAB_SIZE  # index 20

# Extended alphabet (20 AA + mask)
EXTENDED_VOCAB_SIZE = VOCAB_SIZE + 1


def encode_sequence(sequence: str, mask_char: str = "<mask>") -> np.ndarray:
    """
    Encode a protein sequence string into integer indices.

    Args:
        sequence: Protein sequence string. Masked positions should use mask_char.
        mask_char: Character/token representing masked positions.

    Returns:
        np.ndarray of shape (L,) with integer indices.
        Standard AA → 0-19, mask → 20
    """
    encoded = []
    i = 0
    while i < len(sequence):
        # Check for mask token
        if sequence[i:].startswith(mask_char):
            encoded.append(MASK_IDX)
            i += len(mask_char)
        elif sequence[i] in AA_TO_IDX:
            encoded.append(AA_TO_IDX[sequence[i]])
            i += 1
        else:
            raise ValueError(f"Unknown character '{sequence[i]}' at position {i}")
    return np.array(encoded, dtype=np.int64)


def decode_sequence(encoded: np.ndarray, mask_char: str = "?") -> str:
    """
    Decode integer-encoded sequence back to string.

    Args:
        encoded: np.ndarray of integer indices.
        mask_char: Character to use for masked positions.

    Returns:
        Protein sequence string.
    """
    result = []
    for idx in encoded:
        idx = int(idx)
        if idx == MASK_IDX:
            result.append(mask_char)
        elif idx in IDX_TO_AA:
            result.append(IDX_TO_AA[idx])
        else:
            raise ValueError(f"Unknown index {idx}")
    return "".join(result)


def one_hot_encode(sequence: np.ndarray, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """
    Convert integer-encoded sequence to one-hot encoding.
    Masked positions get a zero vector.

    Args:
        sequence: np.ndarray of shape (L,) with integer indices.
        vocab_size: Size of the vocabulary (default: 20 for amino acids).

    Returns:
        np.ndarray of shape (L, vocab_size) one-hot encoding.
    """
    L = len(sequence)
    one_hot = np.zeros((L, vocab_size), dtype=np.float32)
    for i, idx in enumerate(sequence):
        if idx < vocab_size:  # not masked
            one_hot[i, idx] = 1.0
    return one_hot


def one_hot_encode_torch(
    sequence: torch.Tensor, vocab_size: int = VOCAB_SIZE
) -> torch.Tensor:
    """
    Convert integer-encoded sequence to one-hot encoding (PyTorch).

    Args:
        sequence: torch.Tensor of shape (L,) or (B, L).
        vocab_size: Size of the vocabulary.

    Returns:
        torch.Tensor of shape (L, vocab_size) or (B, L, vocab_size).
    """
    # Clamp mask indices to 0 for one-hot, then zero out
    mask = sequence >= vocab_size
    clamped = sequence.clamp(0, vocab_size - 1)
    one_hot = torch.nn.functional.one_hot(clamped, num_classes=vocab_size).float()
    # Zero out masked positions
    if mask.dim() == 1:
        one_hot[mask] = 0.0
    else:
        one_hot[mask] = 0.0
    return one_hot


def create_masked_sequence(
    wt_sequence: str,
    designable_positions: List[int],
    fixed_positions: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Create a partially masked sequence: designable positions are masked,
    non-designable positions retain their wild-type amino acid.

    Args:
        wt_sequence: Wild-type sequence string.
        designable_positions: 0-indexed positions to mask (designable).
        fixed_positions: Positions within designable region to keep fixed (optional).

    Returns:
        np.ndarray of shape (L,) with masked positions set to MASK_IDX.
    """
    encoded = encode_sequence(wt_sequence)
    mask_positions = set(designable_positions)
    if fixed_positions:
        mask_positions -= set(fixed_positions)
    for pos in mask_positions:
        encoded[pos] = MASK_IDX
    return encoded


def get_masked_positions(sequence: np.ndarray) -> np.ndarray:
    """Return indices of masked positions in the sequence."""
    return np.where(sequence == MASK_IDX)[0]


def get_unmasked_positions(sequence: np.ndarray) -> np.ndarray:
    """Return indices of unmasked positions in the sequence."""
    return np.where(sequence != MASK_IDX)[0]


def sequences_to_fasta(
    sequences: List[str],
    output_path: str,
    names: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
):
    """
    Write sequences to a FASTA file.

    Args:
        sequences: List of protein sequence strings.
        output_path: Path to output FASTA file.
        names: Optional list of sequence names.
        scores: Optional list of scores to include in headers.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            name = names[i] if names else f"seq_{i:04d}"
            header = f">{name}"
            if scores is not None:
                header += f" score={scores[i]:.4f}"
            f.write(f"{header}\n{seq}\n")


def load_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Load sequences from a FASTA file.

    Returns:
        List of (name, sequence) tuples.
    """
    sequences = []
    current_name = None
    current_seq = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    sequences.append((current_name, "".join(current_seq)))
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name is not None:
            sequences.append((current_name, "".join(current_seq)))
    return sequences


def pairwise_identity(seq1: str, seq2: str) -> float:
    """
    Compute pairwise sequence identity between two sequences.
    Sequences must be the same length.
    """
    assert len(seq1) == len(seq2), "Sequences must be the same length"
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


def mutation_count(seq: str, wt_seq: str) -> int:
    """Count the number of mutations from wild-type."""
    assert len(seq) == len(wt_seq), "Sequences must be the same length"
    return sum(a != b for a, b in zip(seq, wt_seq))


def compute_diversity(sequences: List[str]) -> float:
    """
    Compute average pairwise diversity (1 - identity) of a set of sequences.
    """
    if len(sequences) < 2:
        return 0.0
    total_identity = 0.0
    count = 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            total_identity += pairwise_identity(sequences[i], sequences[j])
            count += 1
    return 1.0 - (total_identity / count)
