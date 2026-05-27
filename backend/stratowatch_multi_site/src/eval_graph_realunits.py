from __future__ import annotations
import os
import json
from random import sample
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.graph_st_transformer import GraphSTTransformer
from src.utils.metrics import masked_mae, masked_rmse


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inverse_scale(y, mean, std):
    return y * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def main():

    project_root = os.path.dirname(os.path.dirname(__file__))

    NPZ_PATH = os.path.join(
        project_root,
        "data",
        "processed",
        "splits_final_Yscaled_Tin24_Tout6_stride1.npz",
    )

    CKPT_PATH = os.path.join(
        project_root,
        "outputs",
        "checkpoints",
        "graph_st_best.pt",
    )

    SCALER_PATH = os.path.join(project_root, "configs", "target_scaler.json")

    device = get_device()
    print("Device:", device)

    # Load scaler
    with open(SCALER_PATH, "r") as f:
        scaler = json.load(f)

    mean = torch.tensor(scaler["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(scaler["std"], dtype=torch.float32, device=device)

    test_ds = MultisiteDataset(NPZ_PATH, split="test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Load RAW site coordinates (unscaled)
    coords_path = os.path.join(project_root, "configs", "site_coords_raw.npy")
    site_coords = torch.tensor(np.load(coords_path), dtype=torch.float32, device=device)  # (S,2)

    num_features = test_ds.X.shape[-1]
    num_sites = test_ds.X.shape[2]
    tin = test_ds.tin
    tout = test_ds.tout

    model = GraphSTTransformer(
        num_features=num_features,
        tin=tin,
        tout=tout,
        num_sites=num_sites,
    ).to(device)

    

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mae_sum = 0
    rmse_sum = 0
    n = 0

    with torch.no_grad():
        for X, Y, X_mask, Y_mask in test_loader:

            X = X.to(device)
            Y = Y.to(device)
            Y_mask = Y_mask.to(device)

            A_path = os.path.join(project_root, "data", "processed", "adjacency_final.npy")
            A = torch.tensor(np.load(A_path), dtype=torch.float32, device=device)

            Y_hat = model(X, A, site_coords)

            Y_hat_real = inverse_scale(Y_hat, mean, std)
            Y_real = inverse_scale(Y, mean, std)

            mae_sum += masked_mae(Y_hat_real, Y_real, Y_mask).item()
            rmse_sum += masked_rmse(Y_hat_real, Y_real, Y_mask).item()
            n += 1

    print("\nGraph-ST TEST results (REAL units):")
    print("MAE :", mae_sum / n)
    print("RMSE:", rmse_sum / n)


if __name__ == "__main__":
    main()