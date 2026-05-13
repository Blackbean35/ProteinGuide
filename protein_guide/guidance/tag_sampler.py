"""
Taylor-Approximate Guidance (TAG) sampler.

TAG is a fast approximate guidance algorithm that uses gradient-based
Taylor expansion to approximate likelihood ratios, avoiding the need
to evaluate the predictor for every amino acid at every position.

Algorithm:
1. Start from fully masked sequence
2. For each time step t from 0 to 1 (Euler integration):
   a. Get rate matrix from generative model
   b. Compute gradient of log p(y|x_t) w.r.t. input
   c. Use 1st-order Taylor expansion to approximate log-likelihood ratios
   d. Modify rates by the approximate guidance term
   e. Sample transitions (mask → amino acid)
3. Return the decoded sequence

Reference: ProteinGuide paper, Equation S14 and Section 1.5
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from ..models.base_model import BaseGenerativeModel, _softmax
from ..predictors.base_predictor import BasePredictor
from ..data.sequence_utils import (
    VOCAB_SIZE, MASK_IDX, decode_sequence, create_masked_sequence,
    one_hot_encode_torch,
)
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TAGSampler:
    """
    Taylor-Approximate Guidance sampler for flow-matching style generation.

    Uses Euler integration over continuous time with gradient-based
    approximate guidance. Much faster than DEG (O(1) predictor calls
    per time step vs O(S) per position).
    """

    def __init__(
        self,
        generative_model: BaseGenerativeModel,
        predictor,  # Must be a torch.nn.Module-based predictor
        gamma: float = 10.0,
        temperature: float = 0.7,
        dt: float = 0.01,
        wt_weight: float = 0.0,
        stochasticity: float = 0.0,
    ):
        """
        Args:
            generative_model: Pre-trained generative model.
            predictor: Torch-based predictor with differentiable forward pass.
            gamma: Guidance strength.
            temperature: Sampling temperature.
            dt: Euler integration step size.
            wt_weight: Wild-type bias.
            stochasticity: Sampling stochasticity η (for flow matching).
        """
        self.gen_model = generative_model
        self.predictor = predictor
        self.gamma = gamma
        self.temperature = temperature
        self.dt = dt
        self.wt_weight = wt_weight
        self.stochasticity = stochasticity

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
        Generate guided sequences using TAG with Euler integration.

        Returns:
            List of dicts with 'sequence', 'predicted_score', 'n_mutations'.
        """
        actual_design = set(designable_positions)
        if fixed_positions:
            actual_design -= set(fixed_positions)
        actual_design = sorted(actual_design)
        D = len(actual_design)

        n_steps = int(1.0 / self.dt)
        logger.info(
            f"TAG sampling: {n_samples} sequences, {D} positions, "
            f"{n_steps} steps, γ={self.gamma}, T={self.temperature}"
        )

        results = []
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="TAG Sampling")

        for sample_idx in iterator:
            seq = create_masked_sequence(
                wt_sequence, designable_positions, fixed_positions
            )

            # Euler integration from t=0 (fully masked) to t=1 (fully decoded)
            for step in range(n_steps):
                t = step * self.dt
                seq = self._euler_step(
                    seq, t, structure_data, wt_sequence, actual_design
                )

            # Force-decode any remaining masked positions (greedy)
            seq = self._force_decode_remaining(
                seq, structure_data, actual_design
            )

            seq_str = decode_sequence(seq)
            score = self.predictor.predict(seq)
            n_mut = sum(a != b for a, b in zip(seq_str, wt_sequence))
            results.append({
                "sequence": seq_str,
                "predicted_score": score,
                "n_mutations": n_mut,
            })

        scores = [r["predicted_score"] for r in results]
        logger.info(
            f"TAG complete: mean score={np.mean(scores):.3f}, "
            f"active rate={np.mean([s > 0.5 for s in scores]):.1%}"
        )
        return results

    def _euler_step(
        self,
        sequence: np.ndarray,
        t: float,
        structure_data: Dict[str, np.ndarray],
        wt_sequence: str,
        designable_positions: List[int],
    ) -> np.ndarray:
        """
        One Euler integration step for flow-matching with TAG guidance.

        Computes the unmasking rate for each masked position and each amino acid,
        modifies rates with the TAG gradient approximation, and samples transitions.
        """
        # Find currently masked positions
        masked_pos = [p for p in designable_positions if sequence[p] == MASK_IDX]
        if len(masked_pos) == 0:
            return sequence

        # Get generative model logits
        logits = self.gen_model.forward(sequence, structure_data, temperature=1.0)
        if self.wt_weight > 0:
            logits = self.gen_model.apply_wt_weight(
                logits, wt_sequence, self.wt_weight, designable_positions
            )

        # Compute masking flow matching rates
        # Rate of unmasking position i to state s:
        # R(mask→s) = p_θ(s|x_t) / (1 - t) for masked positions
        rate_scale = 1.0 / max(1.0 - t, 1e-6)

        # Compute TAG gradient for guidance
        grad = self._compute_tag_gradient(sequence, designable_positions)

        # For each masked position, compute guided transition rates
        seq = sequence.copy()
        for pos in masked_pos:
            pos_logits = logits[pos] / max(self.temperature, 1e-8)
            p_gen = _softmax(pos_logits)  # (20,)

            # TAG: approximate log-likelihood ratio using gradient
            guided_rates = np.zeros(VOCAB_SIZE)
            for s in range(VOCAB_SIZE):
                base_rate = p_gen[s] * rate_scale

                # Taylor approximation of log p(y|x̃) - log p(y|x)
                if grad is not None and pos < len(grad):
                    # δ = one_hot(s) - current_encoding
                    # For masked position: current = zero vector
                    log_ratio = self.gamma * grad[pos, s]
                    guided_rates[s] = base_rate * np.exp(log_ratio)
                else:
                    guided_rates[s] = base_rate

            # Probability of transition in this time step
            total_rate = np.sum(guided_rates)
            p_transition = min(total_rate * self.dt, 0.99)

            # Sample: transition or stay masked
            if np.random.random() < p_transition:
                # Sample which amino acid
                trans_probs = guided_rates / (total_rate + 1e-30)
                sampled_aa = np.random.choice(VOCAB_SIZE, p=trans_probs)
                seq[pos] = sampled_aa

        return seq

    def _compute_tag_gradient(
        self, sequence: np.ndarray, designable_positions: List[int]
    ) -> Optional[np.ndarray]:
        """
        Compute the gradient of log p(y|x_t) with respect to the input encoding.

        This gradient is used for the Taylor expansion in TAG:
        log p(y|x̃) ≈ log p(y|x) + ∇log p(y|x) · (x̃ - x)

        Returns:
            Gradient array of shape (L, V) or None if not differentiable.
        """
        try:
            # Build differentiable input
            device = torch.device("cpu")
            if hasattr(self.predictor, '_device'):
                device = self.predictor._device

            seq_tensor = torch.tensor(sequence, dtype=torch.long, device=device)
            x_onehot = one_hot_encode_torch(seq_tensor, VOCAB_SIZE + 1).float()
            x_onehot.requires_grad_(True)

            # Forward pass through predictor (needs torch-based predictor)
            if hasattr(self.predictor, 'model'):
                # Extract designable positions for the predictor
                local_x = x_onehot[designable_positions]  # (D, V)
                local_x = local_x.unsqueeze(0)  # (1, D, V)
                logit = self.predictor.model(local_x)
                log_prob = torch.log_sigmoid(logit)
                log_prob.backward()

                # Map gradients back to full sequence
                grad = x_onehot.grad
                if grad is not None:
                    return grad.detach().cpu().numpy()[:, :VOCAB_SIZE]

            return None

        except Exception as e:
            logger.debug(f"TAG gradient computation failed: {e}")
            return None

    def _force_decode_remaining(
        self, sequence: np.ndarray, structure_data, designable_positions
    ) -> np.ndarray:
        """Greedily decode any positions still masked after Euler integration."""
        seq = sequence.copy()
        for pos in designable_positions:
            if seq[pos] == MASK_IDX:
                logits = self.gen_model.forward(seq, structure_data, 1.0)
                pos_logits = logits[pos] / max(self.temperature, 1e-8)
                probs = _softmax(pos_logits)
                seq[pos] = np.random.choice(VOCAB_SIZE, p=probs)
        return seq


def tag_generate(
    gen_model, predictor, structure_data, wt_sequence,
    designable_positions, fixed_positions=None,
    n_samples=100, gamma=10.0, temperature=0.7, dt=0.01, wt_weight=0.0,
) -> List[Dict]:
    """Convenience function for TAG guided generation."""
    sampler = TAGSampler(
        gen_model, predictor, gamma, temperature, dt, wt_weight
    )
    return sampler.sample(
        structure_data, wt_sequence, designable_positions,
        fixed_positions, n_samples,
    )
