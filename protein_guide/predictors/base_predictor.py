"""
Abstract base class for property predictors used in ProteinGuide guidance.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional


class BasePredictor(ABC):
    """
    Interface for property predictors used to guide generation.
    
    Predictors must work on both clean and noisy (partially masked) inputs.
    """

    @abstractmethod
    def predict(self, sequence: np.ndarray) -> float:
        """Predict property probability/value for a (possibly masked) sequence."""
        pass

    @abstractmethod
    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Predict for a batch of sequences. Shape (N, L) -> (N,)."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model weights."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model weights."""
        pass
