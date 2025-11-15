"""PyTorch Dataset class for Parkinson's data."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class ParkinsonsDataset(Dataset):
    """
    PyTorch Dataset for Parkinson's telemonitoring data.

    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples, n_targets)
        device: Device to store tensors on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device: str = "cpu"
    ):
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, targets)
        """
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self.X.shape[1]

    @property
    def n_targets(self) -> int:
        """Number of output targets."""
        return self.y.shape[1] if len(self.y.shape) > 1 else 1
