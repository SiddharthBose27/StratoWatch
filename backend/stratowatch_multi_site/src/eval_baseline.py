"""
src/eval_baseline.py

Evaluate best baseline checkpoint on test split.
"""

from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.st_transformer import STTransformer
from src.utils.metrics import masked_mae, masked_rmse


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    npz_path = os.path.join(project_root, "data", "processed", "splits_final_Tin24_Tout6_stride1.npz")
    ckpt_path = os.path.join(project_root, "outputs", "checkpoints", "st_transformer_best.pt")

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

    mae_sum, rmse_sum, n = 0.0, 0.0, 0

    with torch.no_grad():
        for X, Y, X_mask, Y_mask in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            Y_mask = Y_mask.to(device)

            Y_hat, _ = model(X)

            mae_sum += float(masked_mae(Y_hat, Y, Y_mask).item())
            rmse_sum += float(masked_rmse(Y_hat, Y, Y_mask).item())
            n += 1

    print("\nTEST results:")
    print("MAE :", mae_sum / n)
    print("RMSE:", rmse_sum / n)


if __name__ == "__main__":
    main()