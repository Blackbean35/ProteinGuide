"""
ESM3 wrapper for ProteinGuide framework.

ESM3 is used as a masked language model / inverse folding model.
It can condition on backbone structure to generate sequences.

Requires: pip install esm
Model weights: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from .base_model import BaseGenerativeModel, _softmax
from ..data.sequence_utils import (
    AA_ALPHABET,
    AA_TO_IDX,
    IDX_TO_AA,
    MASK_IDX,
    VOCAB_SIZE,
    decode_sequence,
    create_masked_sequence,
)
import logging

logger = logging.getLogger(__name__)


class ESM3Model(BaseGenerativeModel):
    """
    ESM3 wrapper for inverse folding and guided generation.

    Wraps the EvolutionaryScale ESM3 open model to:
    1. Accept structure input for inverse folding
    2. Return per-position logits for masked positions
    3. Support iterative masked decoding
    4. Handle homo-oligomer logit averaging
    """

    def __init__(
        self,
        model_name: str = "esm3-sm-open-v1",
        device_str: str = "cuda",
        n_chains: int = 1,
    ):
        """
        Args:
            model_name: ESM3 model identifier.
            device_str: Device to load model on ('cuda' or 'cpu').
            n_chains: Number of identical chains (for homo-oligomers).
                      If > 1, logits are averaged across chains.
        """
        self._device = torch.device(device_str)
        self.n_chains = n_chains
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        logger.info(f"ESM3 model will be loaded on {device_str} (lazy loading)")

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        try:
            from esm.models.esm3 import ESM3

            logger.info(f"Loading ESM3 model: {self.model_name}")
            self._model = ESM3.from_pretrained(self.model_name).to(self._device)
            self._model.eval()
            logger.info("ESM3 model loaded successfully")
        except ImportError:
            raise ImportError(
                "ESM3 package not installed. Run: pip install esm\n"
                "And accept the license at: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"
            )

    def forward(
        self,
        sequence: np.ndarray,
        structure_data: Dict[str, np.ndarray],
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Forward pass through ESM3.

        Given a partially masked sequence and structure conditioning,
        returns logits over the 20 amino acid vocabulary for each position.

        For homo-oligomers (n_chains > 1), the input is duplicated and
        logits are averaged across chains (following ProteinGuide paper).
        """
        self._ensure_loaded()

        L_monomer = len(sequence)

        # Build ESM3 input
        protein_input = self._build_protein_input(sequence, structure_data)

        with torch.no_grad():
            output = self._model.forward(
                sequence_tokens=protein_input["sequence_tokens"].to(self._device),
                structure_tokens=protein_input.get("structure_tokens", None),
            )

        # Extract sequence logits: shape (1, L_total, vocab)
        seq_logits = output.sequence_logits[0]  # (L_total, vocab)

        # Map ESM3 vocab to our 20-AA vocab
        logits_20 = self._map_logits_to_aa(seq_logits)  # (L_total, 20)

        # For homo-oligomers: average logits across chains
        if self.n_chains > 1:
            logits_chains = logits_20.reshape(self.n_chains, L_monomer, 20)
            logits_20 = logits_chains.mean(axis=0)  # (L_monomer, 20)
        else:
            logits_20 = logits_20[:L_monomer]

        return logits_20

    def sample_unguided(
        self,
        structure_data: Dict[str, np.ndarray],
        wt_sequence: str,
        designable_positions: List[int],
        fixed_positions: Optional[List[int]] = None,
        n_samples: int = 1,
        temperature: float = 0.5,
        wt_weight: float = 0.0,
        n_decoding_steps: Optional[int] = None,
        stochasticity: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate unguided sequences using iterative masked decoding.

        The decoding proceeds from fully masked to fully unmasked over
        n_decoding_steps steps, unmasking positions by confidence.
        """
        self._ensure_loaded()

        L = len(wt_sequence)
        actual_design_pos = set(designable_positions)
        if fixed_positions:
            actual_design_pos -= set(fixed_positions)
        actual_design_pos = sorted(actual_design_pos)
        D = len(actual_design_pos)

        if n_decoding_steps is None:
            n_decoding_steps = D  # one position per step

        generated = []
        for sample_idx in range(n_samples):
            # Initialize: mask designable positions
            seq = create_masked_sequence(wt_sequence, designable_positions, fixed_positions)

            # Iterative decoding: unmask positions gradually
            positions_to_unmask = list(actual_design_pos)
            np.random.shuffle(positions_to_unmask)

            # Calculate how many positions to unmask per step
            positions_per_step = max(1, D // n_decoding_steps)

            for step in range(0, D, positions_per_step):
                batch_positions = positions_to_unmask[step : step + positions_per_step]

                logits = self.forward(seq, structure_data, temperature=1.0)

                if wt_weight > 0:
                    logits = self.apply_wt_weight(
                        logits, wt_sequence, wt_weight, actual_design_pos
                    )

                for pos in batch_positions:
                    pos_logits = logits[pos] / max(temperature, 1e-8)
                    probs = _softmax(pos_logits)
                    sampled_aa = np.random.choice(VOCAB_SIZE, p=probs)
                    seq[pos] = sampled_aa

            generated.append(decode_sequence(seq))

            if (sample_idx + 1) % 10 == 0:
                logger.info(f"  Generated {sample_idx + 1}/{n_samples} sequences")

        return generated

    def _build_protein_input(
        self, sequence: np.ndarray, structure_data: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Build tokenized input for ESM3.

        Converts our internal sequence representation and structure data
        into ESM3-compatible token tensors.
        """
        try:
            from esm.utils.structure.protein_chain import ProteinChain
            from esm.tokenization import get_model_tokenizers

            # Build sequence string for ESM3
            seq_str = ""
            for idx in sequence:
                if idx == MASK_IDX:
                    seq_str += "_"  # ESM3 mask character
                elif idx in IDX_TO_AA:
                    seq_str += IDX_TO_AA[idx]
                else:
                    seq_str += "_"

            # Handle homo-oligomer
            if self.n_chains > 1:
                seq_str = seq_str * self.n_chains

            # Tokenize sequence
            tokenizers = get_model_tokenizers(self.model_name)
            seq_tokens = tokenizers.sequence.encode(seq_str)
            seq_tensor = torch.tensor([seq_tokens], dtype=torch.long)

            result = {"sequence_tokens": seq_tensor}

            # Add structure tokens if structure data is available
            if structure_data is not None:
                try:
                    coords = np.stack(
                        [structure_data["N"], structure_data["CA"],
                         structure_data["C"], structure_data["O"]],
                        axis=1,
                    )
                    chain = ProteinChain.from_backbone_atom_coordinates(
                        coords, sequence=structure_data.get("sequence", seq_str)
                    )
                    struct_tokens = chain.to_structure_tokens(tokenizers.structure)

                    if self.n_chains > 1:
                        struct_tokens = struct_tokens.repeat(1, self.n_chains)

                    result["structure_tokens"] = struct_tokens.to(self._device)
                except Exception as e:
                    logger.warning(f"Could not encode structure tokens: {e}")

            return result

        except ImportError:
            # Fallback: simple tokenization without structure
            logger.warning("Full ESM3 tokenization not available, using simple encoding")
            from ..data.sequence_utils import encode_sequence

            seq_tensor = torch.tensor([sequence], dtype=torch.long)
            return {"sequence_tokens": seq_tensor}

    def _map_logits_to_aa(self, esm_logits: torch.Tensor) -> np.ndarray:
        """
        Map ESM3 vocabulary logits to our standard 20-AA logits.

        ESM3 uses its own tokenizer; we extract the logits corresponding
        to the standard 20 amino acids.
        """
        try:
            from esm.tokenization import get_model_tokenizers

            tokenizers = get_model_tokenizers(self.model_name)
            logits_np = esm_logits.cpu().numpy()
            L = logits_np.shape[0]
            result = np.zeros((L, VOCAB_SIZE), dtype=np.float32)

            for aa, idx in AA_TO_IDX.items():
                token_id = tokenizers.sequence.encode(aa)
                if isinstance(token_id, list) and len(token_id) > 0:
                    # Token ID might include BOS/EOS, take the actual token
                    if len(token_id) == 3:  # BOS + token + EOS
                        token_id = token_id[1]
                    elif len(token_id) == 1:
                        token_id = token_id[0]
                    else:
                        token_id = token_id[0]
                result[:, idx] = logits_np[:, token_id]

            return result

        except Exception:
            # Fallback: assume first 20 logits correspond to AA
            logits_np = esm_logits.cpu().numpy()
            return logits_np[:, :VOCAB_SIZE]

    @property
    def device(self) -> torch.device:
        return self._device
