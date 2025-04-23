import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple

class SequenceDataset(Dataset):
    """Creates sequences from time-series data for RNN input."""
    def __init__(self, features: np.ndarray, targets: pd.Series, sequence_length: int):
        """Initialize the dataset.

        Args:
            features (np.ndarray): The input features (scaled), shape (num_samples, num_features).
            targets (pd.Series): The target variable, shape (num_samples,).
            sequence_length (int): The length of the sequences to create.
        """
        if not isinstance(features, np.ndarray):
             raise TypeError(f"Expected features to be np.ndarray, got {type(features)}")
        # Convert targets to numpy if it's a pandas Series for consistent indexing
        if isinstance(targets, pd.Series):
            targets = targets.to_numpy()
        elif not isinstance(targets, np.ndarray):
             raise TypeError(f"Expected targets to be pd.Series or np.ndarray, got {type(targets)}")

        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same number of samples.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        # Add validation for sequence_length vs num_samples
        if sequence_length > len(features):
            raise ValueError(f"sequence_length ({sequence_length}) cannot be greater than the number of samples ({len(features)}).")

        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        # The number of samples will be reduced because we need `sequence_length` history
        # for each sample.
        self.num_effective_samples = len(features) - sequence_length + 1

        print(f"Created SequenceDataset: {self.num_effective_samples} samples, seq_len={sequence_length}, features_per_step={features.shape[1]}")


    def __len__(self) -> int:
        """Return the number of sequences we can create."""
        return self.num_effective_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence sample.

        Args:
            idx (int): The index of the *end* of the sequence.
                       We use the index relative to the *effective* number of samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The input sequence (sequence_length, num_features)
                - The target value corresponding to the end of the sequence (1,)
        """
        if idx < 0 or idx >= self.num_effective_samples:
            raise IndexError(f"Index {idx} out of bounds for effective samples {self.num_effective_samples}")

        # Calculate the actual start and end index in the original data
        start_idx = idx
        end_idx = idx + self.sequence_length # Slice end index is exclusive
        target_idx = end_idx - 1 # Target corresponds to the last element of the sequence

        # Extract the sequence of features
        sequence_features = self.features[start_idx:end_idx, :]
        # Extract the target value
        target_value = self.targets[target_idx]

        # Convert to PyTorch tensors
        # Ensure target is treated as a regression value (float)
        # Reshape target to be (1,) or ensure it's treated as a scalar tensor suitable for loss calc
        return torch.tensor(sequence_features, dtype=torch.float32), torch.tensor(target_value, dtype=torch.float32).unsqueeze(0) 