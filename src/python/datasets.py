import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TabularDataset(Dataset):
    """Custom Dataset for tabular data.

    Converts pandas DataFrame/Series to PyTorch tensors.
    Assumes features (X) and target (y) are already preprocessed 
    (numeric, scaled if necessary).
    """
    def __init__(self, features, target, feature_dtype=torch.float32, target_dtype=torch.float32):
        
        # Modified check: Expect features to be a NumPy array now
        if not isinstance(features, np.ndarray):
            raise TypeError("Features must be a NumPy array.")
        # Keep target check as pandas Series
        if not isinstance(target, pd.Series):
            raise TypeError("Target must be a pandas Series.")
        if features.shape[0] != len(target):
            raise ValueError("Features and target must have the same number of samples.")

        # Create tensor directly from NumPy array (no .values needed)
        self.features = torch.tensor(features, dtype=feature_dtype)
        # Keep .values for pandas Series target
        # Reshape target to be [n_samples, 1] as typically expected by loss functions
        self.target = torch.tensor(target.values, dtype=target_dtype).unsqueeze(1)
        
        print(f"Created TabularDataset: {len(self)} samples, {self.features.shape[1]} features.")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns features and target for a given index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return self.features[idx], self.target[idx] 