"""
src/train_baseline.py

Train the baseline STTransformer on the final processed dataset.

- Uses MPS on Mac if available
- Masked MAE loss (stable)
- Saves best model by validation MAE
"""

from __future__ import annotations

import os
import time
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.st_transformer import STTransformer
from src.utils.metrics import masked_mae, masked_rmse, masked_huber


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    npz_path = os.path.join(project_root, "data", "processed", "splits_final_Yscaled_Tin24_Tout6_stride1.npz")
    out_dir = os.path.join(project_root, "outputs", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Hyperparams (safe for M1 8GB) ----
    batch_size = 32
    lr = 3e-4
    epochs = 10  # start small; we can increase later
    d_model = 128
    nhead = 4
    num_layers_time = 2
    num_layers_space = 2
    dropout = 0.1

    device = get_device()
    print("Device:", device)

    # ---- Data ----
    train_ds = MultisiteDataset(npz_path, split="train")
    val_ds = MultisiteDataset(npz_path, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Model ----
    num_features = train_ds.X.shape[-1]
    num_sites = train_ds.X.shape[2]
    tin = train_ds.tin
    tout = train_ds.tout

    model = STTransformer(
        num_features=num_features,
        tin=tin,
        tout=tout,
        num_sites=num_sites,
        d_model=d_model,
        nhead=nhead,
        num_layers_time=num_layers_time,
        num_layers_space=num_layers_space,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    best_path = os.path.join(out_dir, "st_transformer_best.pt")

    patience = 3
    bad_epochs = 0

    # ---- Train loop ----
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        train_mae_sum = 0.0
        train_batches = 0

        for X, Y, X_mask, Y_mask in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            Y_mask = Y_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            Y_hat, _ = model(X)

            loss = masked_huber(Y_hat, Y, Y_mask, delta=1.0)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_mae_sum += float(loss.item())
            train_batches += 1

        train_mae = train_mae_sum / max(train_batches, 1)

        # ---- Validation ----
        model.eval()
        val_mae_sum = 0.0
        val_rmse_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for X, Y, X_mask, Y_mask in val_loader:
                X = X.to(device)
                Y = Y.to(device)
                Y_mask = Y_mask.to(device)

                Y_hat, _ = model(X)
                val_mae_sum += float(masked_mae(Y_hat, Y, Y_mask).item())
                val_rmse_sum += float(masked_rmse(Y_hat, Y, Y_mask).item())
                val_batches += 1

        val_mae = val_mae_sum / max(val_batches, 1)
        val_rmse = val_rmse_sum / max(val_batches, 1)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | train_MAE={train_mae:.4f} | val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f} | {dt:.1f}s")

        

        # Early stopping + Save best
        if val_mae < best_val:
            best_val = val_mae
            bad_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                     "epoch": epoch,
                    "val_mae": val_mae,
                    "feature_cols": train_ds.feature_cols,
                    "target_cols": train_ds.target_cols,
                    "tin": tin,
                    "tout": tout,
                    "num_sites": num_sites,
                    "num_features": num_features,
                },
                best_path,
            )
            print("✅ Saved best checkpoint:", best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    print("\nDone. Best val MAE:", best_val)


if __name__ == "__main__":
    main()