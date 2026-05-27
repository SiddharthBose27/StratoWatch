"""
src/data/dataset.py

Loads final preprocessed splits into PyTorch Dataset.

Expected file:
data/processed/splits_final_Tin24_Tout6_stride1.npz
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MultisiteDataset(Dataset):
    def __init__(self, npz_path: str, split: str):
        """
        split: "train", "val", or "test"
        """

        data = np.load(npz_path, allow_pickle=True)

        self.X = torch.tensor(data[f"X_{split}"], dtype=torch.float32)
        self.Y = torch.tensor(data[f"Y_{split}"], dtype=torch.float32)
        self.X_mask = torch.tensor(data[f"X_mask_{split}"], dtype=torch.float32)
        self.Y_mask = torch.tensor(data[f"Y_mask_{split}"], dtype=torch.float32)

        self.feature_cols = list(data["feature_cols"])
        self.target_cols = list(data["target_cols"])
        # tin/tout may be stored as 0-d arrays or 1-element arrays inside npz
        self.tin = int(np.array(data["tin"]).reshape(-1)[0])
        self.tout = int(np.array(data["tout"]).reshape(-1)[0])
        self.sites = list(data["sites"])

        print(f"{split.upper()} loaded:")
        print("X:", self.X.shape)
        print("Y:", self.Y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.Y[idx],
            self.X_mask[idx],
            self.Y_mask[idx],
        )