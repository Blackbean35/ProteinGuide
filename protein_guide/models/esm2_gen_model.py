"""
ESM2-based generative model for ProteinGuide.

ESM2 is a masked language model (MLM) — it predicts the probability of each
amino acid at a masked position given the surrounding context. This is exactly
what ProteinGuide's DEG/TAG guidance algorithms need:

    p_gen(x_i | x_{decoded}) = softmax(ESM2_logits at position i)

Advantages over ESM3:
- Completely open access (no HuggingFace gating / license wall)
- Already installed if you ran train_on_tadabench (ESM2 predictor)
- Fast inference, well-supported

Decoding strategy:
- Start from a fully masked sequence (designable positions = <mask>)
- At each step, decode one position by sampling from MLM logits
- Structure conditioning: ESM2 is sequence-only; structure is unused
  (the structure is implicitly encoded in the predictor guidance signal)
"""

import numpy as np
import torch
from typing import Dict, List, Optional

from .base_model import BaseGenerativeModel, _softmax

import logging
logger = logging.getLogger(__name__)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_ALPHABET)}
MASK_IDX = 20


class ESM2GenerativeModel(BaseGenerativeModel):
    """
    ESM2 masked language model used as a sequence generative model.

    The model uses ESM2's masked token predictions as p_gen(x_i | context).
    Structure data is accepted but not used (ESM2 is sequence-only).

    Usage:
        gen_model = ESM2GenerativeModel()
        # Use with DEGSampler or TAGSampler just like ESM3Model
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        device_str: str = "cuda",
        n_chains: int = 1,
    ):
        """
        Args:
            model_name: HuggingFace ESM2 model ID.
                - "facebook/esm2_t6_8M_UR50D"    (8M params, fastest)
                - "facebook/esm2_t12_35M_UR50D"   (35M, good balance) [default]
                - "facebook/esm2_t30_150M_UR50D"  (150M, better quality)
                - "facebook/esm2_t33_650M_UR50D"  (650M, best quality, slow)
            device_str: 'cuda' or 'cpu'.
            n_chains: Ignored (for API compatibility).
        """
        self._device = torch.device(
            "cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        logger.info(
            f"ESM2GenerativeModel: {model_name} will be loaded lazily on {self._device}"
        )

    def _ensure_loaded(self):
        if self._model is None:
            logger.info(f"Loading ESM2 generative model: {self.model_name}")
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(
                self.model_name
            ).to(self._device)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            logger.info(
                f"ESM2 generative model loaded | device={self._device}"
            )

    @property
    def device(self) -> torch.device:
        return self._device

    def _seq_array_to_str(self, sequence: np.ndarray, mask_with_token: bool = True) -> str:
        """
        Convert integer-encoded sequence to string.
        Masked positions (index 20) → '<mask>' token if mask_with_token else 'A'.
        """
        chars = []
        for idx in sequence:
            idx = int(idx)
            if idx == MASK_IDX:
                chars.append("<mask>" if mask_with_token else "A")
            else:
                chars.append(IDX_TO_AA.get(idx, "A"))
        return "".join(chars)

    @torch.no_grad()
    def forward(
        self,
        sequence: np.ndarray,
        structure_data: Dict[str, np.ndarray],
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Compute per-position logits using ESM2 MLM head.

        Masked positions (index 20) are passed as '<mask>' to ESM2.
        Non-masked positions are passed as their amino acid character.

        Args:
            sequence: np.ndarray shape (L,), integer-encoded. Masked = 20.
            structure_data: Ignored (ESM2 is sequence-only).
            temperature: Not applied here; applied by the sampler.

        Returns:
            np.ndarray shape (L, 20) — raw logits over the 20 standard AA.
        """
        self._ensure_loaded()

        L = len(sequence)
        # Build the input string: masked positions → '<mask>'
        # ESM2 tokenizer handles '<mask>' as a special token
        tokens = []
        for idx in sequence:
            idx = int(idx)
            if idx == MASK_IDX:
                tokens.append(self._tokenizer.mask_token)  # '<mask>'
            else:
                tokens.append(IDX_TO_AA.get(idx, "A"))

        # Join as a single sequence string (ESM2 tokenizer handles single chars)
        seq_str = "".join(tokens)

        inputs = self._tokenizer(
            seq_str,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self._device)

        if self._device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self._model(**inputs)
        else:
            out = self._model(**inputs)

        # logits shape: (1, L+2, vocab_size)  (+2 for [CLS] and [EOS])
        logits_full = out.logits[0, 1:-1, :]  # (L, vocab_size)

        # Extract logits for the 20 standard AA tokens
        # ESM2 tokenizer vocab maps each AA to a specific token id
        aa_token_ids = [
            self._tokenizer.convert_tokens_to_ids(aa) for aa in AA_ALPHABET
        ]
        logits_aa = logits_full[:, aa_token_ids].float().cpu().numpy()  # (L, 20)

        return logits_aa

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
        Generate unguided sequences via iterative masked decoding.

        Decodes designable positions one at a time in a random order,
        using ESM2 MLM predictions as the sampling distribution.
        """
        from ..data.sequence_utils import (
            encode_sequence, decode_sequence, create_masked_sequence
        )

        mask_positions = set(designable_positions)
        if fixed_positions:
            mask_positions -= set(fixed_positions)
        decode_order = sorted(mask_positions)

        results = []
        for _ in range(n_samples):
            seq = create_masked_sequence(wt_sequence, designable_positions, fixed_positions)
            order = np.array(decode_order, dtype=int)
            np.random.shuffle(order)

            for pos in order:
                logits = self.forward(seq, structure_data, temperature=1.0)
                pos_logits = logits[pos] / max(temperature, 1e-8)

                if wt_weight > 0:
                    wt_aa = wt_sequence[pos]
                    if wt_aa in AA_TO_IDX:
                        pos_logits[AA_TO_IDX[wt_aa]] += wt_weight

                probs = _softmax(pos_logits)
                sampled = np.random.choice(len(AA_ALPHABET), p=probs)
                seq[pos] = sampled

            results.append(decode_sequence(seq))

        return results
