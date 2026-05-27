"""
src/train_graph.py

Train Graph-Enhanced Spatio-Temporal Transformer
Uses hybrid adjacency matrix.
"""

from __future__ import annotations
import os
from matplotlib.pylab import sample
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.graph_st_transformer import GraphSTTransformer
from src.utils.metrics import masked_mae, masked_rmse


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():

    project_root = os.path.dirname(os.path.dirname(__file__))

    NPZ_PATH = os.path.join(
        project_root,
        "data",
        "processed",
        "splits_final_Yscaled_Tin24_Tout6_stride1.npz",
    )

    ADJ_PATH = os.path.join(project_root, "data", "processed", "adjacency_final.npy")

    OUT_DIR = os.path.join(project_root, "outputs", "checkpoints")
    os.makedirs(OUT_DIR, exist_ok=True)

    device = get_device()
    print("Device:", device)

    # Load adjacency
    A = torch.tensor(np.load(ADJ_PATH), dtype=torch.float32, device=device)

    # Datasets
    train_ds = MultisiteDataset(NPZ_PATH, split="train")
    val_ds = MultisiteDataset(NPZ_PATH, split="val")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    print("TRAIN loaded:", train_ds.X.shape)
    print("VAL loaded:", val_ds.X.shape)

    num_features = train_ds.X.shape[-1]
    num_sites = train_ds.X.shape[2]
    tin = train_ds.tin
    tout = train_ds.tout

    model = GraphSTTransformer(
        num_features=num_features,
        tin=tin,
        tout=tout,
        num_sites=num_sites,
    ).to(device)

    # ------------------------------
    # Load RAW site coordinates (unscaled)
    # ------------------------------
    coords_path = os.path.join(project_root, "configs", "site_coords_raw.npy")
    site_coords = torch.tensor(np.load(coords_path), dtype=torch.float32, device=device)  # (S,2)

    print("\nModel created:")
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()  # Huber loss

    epochs = 30
    best_val = float("inf")
    patience = 4
    bad_epochs = 0

    best_path = os.path.join(OUT_DIR, "graph_st_best.pt")

    for epoch in range(1, epochs + 1):

        # -------- TRAIN --------
        model.train()
        train_mae = 0
        n_train = 0

        for X, Y, X_mask, Y_mask in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            Y_mask = Y_mask.to(device)

            optimizer.zero_grad()

            Y_hat = model(X, A, site_coords)

            Y_hat = Y_hat.contiguous()
            Y = Y.contiguous()

            loss_raw = criterion(Y_hat, Y)
            loss = (loss_raw * Y_mask).sum() / (Y_mask.sum() + 1e-8)
            loss.backward()
            optimizer.step()

            train_mae += masked_mae(Y_hat, Y, Y_mask).item()
            n_train += 1

        train_mae /= n_train

        # -------- VAL --------
        model.eval()
        val_mae = 0
        val_rmse = 0
        n_val = 0

        with torch.no_grad():
            for X, Y, X_mask, Y_mask in val_loader:
                X = X.to(device)
                Y = Y.to(device)
                Y_mask = Y_mask.to(device)

                Y_hat = model(X, A, site_coords)

                val_mae += masked_mae(Y_hat, Y, Y_mask).item()
                val_rmse += masked_rmse(Y_hat, Y, Y_mask).item()
                n_val += 1

        val_mae /= n_val
        val_rmse /= n_val

        print(
            f"Epoch {epoch:02d} | "
            f"train_MAE={train_mae:.4f} | "
            f"val_MAE={val_mae:.4f} | "
            f"val_RMSE={val_rmse:.4f}"
        )

        # -------- EARLY STOPPING --------
        if val_mae < best_val:
            best_val = val_mae
            bad_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_mae": val_mae,
                },
                best_path,
            )
            print("✅ Saved best checkpoint:", best_path)

        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\nDone. Best val MAE:", best_val)


if __name__ == "__main__":
    main()