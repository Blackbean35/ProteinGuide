"""
Discrete-time Exact Guidance (DEG) sampler.

This is the primary guidance algorithm used in the ABE engineering experiment.
It performs exact classifier guidance by evaluating the predictor for every
possible amino acid at each position during autoregressive decoding.

Algorithm:
1. Start from a fully masked sequence (designable positions masked)
2. Sample a random decoding order (permutation of designable positions)
3. For each position in order:
   a. Get generative model logits p_gen(x_i | x_{decoded})
   b. For each amino acid s ∈ {A,C,...,Y}:
      - Set x_i = s temporarily
      - Evaluate predictor: p(y | x_with_s)
   c. Compute guided probability: p_guided(s) ∝ p_gen(s) · p(y|x_with_s)^γ
   d. Sample from guided distribution
4. Return the fully decoded sequence

Reference: ProteinGuide paper, Section 1.4.6 ("Implications for guidance")
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Callable
from ..models.base_model import BaseGenerativeModel, _softmax
from ..predictors.base_predictor import BasePredictor
from ..data.sequence_utils import (
    VOCAB_SIZE,
    MASK_IDX,
    IDX_TO_AA,
    decode_sequence,
    create_masked_sequence,
)
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DEGSampler:
    """
    Discrete-time Exact Guidance sampler for protein sequence generation.

    Performs exact Bayesian guidance during autoregressive (AO-ARM style) decoding.
    At each position, evaluates all possible amino acids and weights them by
    the predictor score raised to the power of gamma (guidance strength).
    """

    def __init__(
        self,
        generative_model: BaseGenerativeModel,
        predictor: BasePredictor,
        gamma: float = 100.0,
        temperature: float = 0.5,
        wt_weight: float = 0.0,
    ):
        """
        Args:
            generative_model: Pre-trained generative model (ESM3 or ProteinMPNN).
            predictor: Trained property predictor (must handle masked inputs).
            gamma: Guidance strength. 1.0 = Bayes' rule, >1.0 = stronger guidance.
            temperature: Sampling temperature for generative model.
            wt_weight: Wild-type bias weight (0 = no bias).
        """
        self.gen_model = generative_model
        self.predictor = predictor
        self.gamma = gamma
        self.temperature = temperature
        self.wt_weight = wt_weight

    def sample(
        self,
        structure_data: Dict[str, np.ndarray],
        wt_sequence: str,
        designable_positions: List[int],
        fixed_positions: Optional[List[int]] = None,
        n_samples: int = 100,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate guided sequences using DEG.

        Args:
            structure_data: Backbone structure from load_pdb_structure().
            wt_sequence: Wild-type sequence string.
            designable_positions: 0-indexed positions to design.
            fixed_positions: Positions to keep at wild-type.
            n_samples: Number of sequences to generate.
            show_progress: Show progress bar.

        Returns:
            List of dicts, each containing:
                - 'sequence': generated sequence string
                - 'predicted_score': predictor score for the sequence
                - 'n_mutations': number of mutations from wild-type
        """
        # Determine actual positions to decode
        actual_design = set(designable_positions)
        if fixed_positions:
            actual_design -= set(fixed_positions)
        actual_design = sorted(actual_design)
        D = len(actual_design)

        logger.info(
            f"DEG sampling: {n_samples} sequences, "
            f"{D} designable positions, γ={self.gamma}, T={self.temperature}"
        )

        results = []
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="DEG Sampling")

        for sample_idx in iterator:
            # Initialize: mask designable positions
            seq = create_masked_sequence(
                wt_sequence, designable_positions, fixed_positions
            )

            # Random decoding order
            decode_order = np.array(actual_design)
            np.random.shuffle(decode_order)

            # Decode position by position with guidance
            for pos in decode_order:
                seq = self._guided_decode_position(
                    seq, pos, structure_data, wt_sequence, actual_design
                )

            # Evaluate final sequence
            seq_str = decode_sequence(seq)
            score = self.predictor.predict(seq)
            n_mut = sum(a != b for a, b in zip(seq_str, wt_sequence))

            results.append({
                "sequence": seq_str,
                "predicted_score": score,
                "n_mutations": n_mut,
            })

        # Summary statistics
        scores = [r["predicted_score"] for r in results]
        logger.info(
            f"DEG complete: mean score={np.mean(scores):.3f}, "
            f"active rate={np.mean([s > 0.5 for s in scores]):.1%}"
        )

        return results

    def _guided_decode_position(
        self,
        sequence: np.ndarray,
        position: int,
        structure_data: Dict[str, np.ndarray],
        wt_sequence: str,
        designable_positions: List[int],
    ) -> np.ndarray:
        """
        Decode a single position with exact guidance.

        For each of the 20 amino acids:
        1. Temporarily place that amino acid at the position
        2. Evaluate the predictor
        3. Weight by generative model probability × predictor^γ
        4. Sample from the weighted distribution
        """
        # Get generative model logits for this position
        logits_all = self.gen_model.forward(
            sequence, structure_data, temperature=1.0
        )

        # Apply wild-type weight if specified
        if self.wt_weight > 0:
            logits_all = self.gen_model.apply_wt_weight(
                logits_all, wt_sequence, self.wt_weight, designable_positions
            )

        # Softmax over amino acids for this position
        pos_logits = logits_all[position] / max(self.temperature, 1e-8)
        p_gen = _softmax(pos_logits)  # (20,)

        # Evaluate predictor for each possible amino acid
        predictor_scores = np.zeros(VOCAB_SIZE)
        for aa_idx in range(VOCAB_SIZE):
            seq_candidate = sequence.copy()
            seq_candidate[position] = aa_idx
            predictor_scores[aa_idx] = self.predictor.predict(seq_candidate)

        # Compute guided probabilities
        # p_guided(s) ∝ p_gen(s) · p(y|x_with_s)^γ
        guided_log = np.log(p_gen + 1e-30) + self.gamma * np.log(
            predictor_scores + 1e-30
        )
        guided_probs = _softmax(guided_log)

        # Sample
        sampled_aa = np.random.choice(VOCAB_SIZE, p=guided_probs)
        sequence[position] = sampled_aa

        return sequence


def deg_generate(
    gen_model: BaseGenerativeModel,
    predictor: BasePredictor,
    structure_data: Dict[str, np.ndarray],
    wt_sequence: str,
    designable_positions: List[int],
    fixed_positions: Optional[List[int]] = None,
    n_samples: int = 100,
    gamma: float = 100.0,
    temperature: float = 0.5,
    wt_weight: float = 0.0,
) -> List[Dict]:
    """
    Convenience function for DEG guided generation.

    See DEGSampler.sample() for details.
    """
    sampler = DEGSampler(
        generative_model=gen_model,
        predictor=predictor,
        gamma=gamma,
        temperature=temperature,
        wt_weight=wt_weight,
    )
    return sampler.sample(
        structure_data=structure_data,
        wt_sequence=wt_sequence,
        designable_positions=designable_positions,
        fixed_positions=fixed_positions,
        n_samples=n_samples,
    )
