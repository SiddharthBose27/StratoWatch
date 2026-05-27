"""
src/plot_predictions.py

Creates heatmaps comparing True vs Pred for:
- O3_target
- NO2_target
for selected horizon (e.g., t+1 and t+6)

Saves figures to outputs/figures/
"""

from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.st_transformer import STTransformer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def heatmap_save(mat, title, outpath, xlabel="Time (samples)", ylabel="Site"):
    plt.figure(figsize=(12, 4))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    npz_path = os.path.join(project_root, "data", "processed", "splits_final_Tin24_Tout6_stride1.npz")
    ckpt_path = os.path.join(project_root, "outputs", "checkpoints", "st_transformer_best.pt")
    fig_dir = os.path.join(project_root, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    device = get_device()
    print("Device:", device)

    test_ds = MultisiteDataset(npz_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    num_features = test_ds.X.shape[-1]
    num_sites = test_ds.X.shape[2]
    tin = test_ds.tin
    tout = test_ds.tout

    model = STTransformer(
        num_features=num_features,
        tin=tin,
        tout=tout,
        num_sites=num_sites,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect a manageable number of predictions for plotting
    # We'll take first N_plot batches
    N_plot = 800  # total samples (windows) to visualize
    all_true = []
    all_pred = []

    with torch.no_grad():
        for X, Y, X_mask, Y_mask in test_loader:
            X = X.to(device)
            Y_hat, _ = model(X)

            all_true.append(Y.cpu().numpy())
            all_pred.append(Y_hat.cpu().numpy())

            if sum(a.shape[0] for a in all_true) >= N_plot:
                break

    Y_true = np.concatenate(all_true, axis=0)[:N_plot]  # (N, Tout, S, 2)
    Y_pred = np.concatenate(all_pred, axis=0)[:N_plot]

    # Choose horizons to visualize
    horizons = [0, tout - 1]  # t+1 and t+Tout
    targets = {0: "O3_target", 1: "NO2_target"}

    for h in horizons:
        for c, name in targets.items():
            # Heatmap expects (S, N) so transpose
            true_mat = Y_true[:, h, :, c].T  # (S, N)
            pred_mat = Y_pred[:, h, :, c].T  # (S, N)
            err_mat = (pred_mat - true_mat)

            heatmap_save(
                true_mat,
                f"TRUE {name} | horizon t+{h+1}",
                os.path.join(fig_dir, f"true_{name}_h{h+1}.png"),
            )
            heatmap_save(
                pred_mat,
                f"PRED {name} | horizon t+{h+1}",
                os.path.join(fig_dir, f"pred_{name}_h{h+1}.png"),
            )
            heatmap_save(
                err_mat,
                f"ERROR (pred-true) {name} | horizon t+{h+1}",
                os.path.join(fig_dir, f"err_{name}_h{h+1}.png"),
            )

    print("\n✅ Saved heatmaps to:", fig_dir)
    print("Files include: true_*, pred_*, err_* for O3 and NO2 at t+1 and t+6")


if __name__ == "__main__":
    main()