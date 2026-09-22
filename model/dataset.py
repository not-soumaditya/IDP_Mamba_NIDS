"""
NIDS Sequence Dataset — Converts tabular network flows into sequences.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class NIDSSequenceDataset(Dataset):
    """
    Converts tabular network flow data into sequences using a sliding window.

    Each sample is a window of `seq_len` consecutive flows.
    The label is the LAST flow's label in the window — we predict the current
    flow based on the context of the previous flows.

    Args:
        dataframe: pandas DataFrame with features and labels
        feature_cols: list of column names to use as input features
        label_col: column name for the target label
        seq_len: number of consecutive flows per sequence (default: 32)
        stride: step size between consecutive windows (default: 1)
    """

    def __init__(self, dataframe, feature_cols, label_col, seq_len=32, stride=1):
        self.seq_len = seq_len
        self.features = dataframe[feature_cols].values.astype(np.float32)
        self.labels = dataframe[label_col].values.astype(np.int64)
        self.stride = stride
        self.valid_indices = list(
            range(0, len(self.features) - seq_len + 1, stride)
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.seq_len
        x = torch.tensor(self.features[start:end])  # (seq_len, num_features)
        y = torch.tensor(self.labels[end - 1])       # label of the last flow
        return x, y
