"""
src/eval_baseline_realunits.py

Evaluate best checkpoint on TEST in REAL units (inverse target scaling).

- Loads target_scaler.json
- Inverse transforms y_hat and y
- Computes MAE and RMSE in original units
"""

from __future__ import annotations
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.st_transformer import STTransformer
from src.utils.metrics import masked_mae, masked_rmse


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inverse_scale(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # y: (B, Tout, S, C)
    return y * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))

    # IMPORTANT: use the Y-scaled dataset file
    npz_path = os.path.join(project_root, "data", "processed", "splits_final_Yscaled_Tin24_Tout6_stride1.npz")
    ckpt_path = os.path.join(project_root, "outputs", "checkpoints", "st_transformer_best.pt")
    scaler_path = os.path.join(project_root, "configs", "target_scaler.json")

    device = get_device()
    print("Device:", device)

    # Load scaler
    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    mean = torch.tensor(scaler["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(scaler["std"], dtype=torch.float32, device=device)
    print("Targets:", scaler["targets"])
    print("Mean:", scaler["mean"])
    print("Std :", scaler["std"])

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

    mae_sum = 0
    rmse_sum = 0

    Y_hat_all = []
    Y_all = []
    Y_mask_all = []

    with torch.no_grad():
        for X, Y, X_mask, Y_mask in test_loader:

            X = X.to(device)
            Y = Y.to(device)
            Y_mask = Y_mask.to(device)

            Y_hat, _ = model(X, X_mask)

            Y_hat_real = inverse_scale(Y_hat, mean, std)
            Y_real = inverse_scale(Y, mean, std)

            Y_hat_all.append(Y_hat_real.cpu())
            Y_all.append(Y_real.cpu())
            Y_mask_all.append(Y_mask.cpu())

    Y_hat_all = torch.cat(Y_hat_all, dim=0)
    Y_all = torch.cat(Y_all, dim=0)
    Y_mask_all = torch.cat(Y_mask_all, dim=0)
    
    # Calculate global MAE and RMSE
    diff = (Y_hat_all - Y_all) * Y_mask_all
    mae = torch.sum(torch.abs(diff)) / (torch.sum(Y_mask_all) + 1e-8)
    rmse = torch.sqrt(torch.sum(diff**2) / (torch.sum(Y_mask_all) + 1e-8))
    
    # Calculate global R2
    # R2 = 1 - (SS_res / SS_tot)
    ss_res = torch.sum((diff)**2)
    
    # Global mean of Y for R2
    y_mean = torch.sum(Y_all * Y_mask_all) / (torch.sum(Y_mask_all) + 1e-8)
    ss_tot = torch.sum(((Y_all - y_mean) * Y_mask_all)**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    print("\nBaseline ST TEST results (REAL units):")
    print(f"MAE : {mae.item():.4f}")
    print(f"RMSE: {rmse.item():.4f}")
    print(f"R2  : {r2.item():.4f}")

    # Persist the exact arrays used for the frozen evaluation.  Graph-ST
    # evaluation already writes the equivalent artifacts; keeping the
    # baseline in the same convention lets downstream plotting consume real
    # predictions rather than reconstructing anything from summary metrics.
    output_dir = os.path.join(project_root, "outputs", "final_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    np.save(
        os.path.join(output_dir, "st_test_predictions_real.npy"),
        Y_hat_all.numpy(),
    )
    np.save(
        os.path.join(output_dir, "st_test_truth_real.npy"),
        Y_all.numpy(),
    )
    np.save(
        os.path.join(output_dir, "st_test_mask.npy"),
        Y_mask_all.numpy(),
    )
    
    # Calculate separate metrics for O3 (idx 0) and NO2 (idx 1)
    target_names = ["O3", "NO2"]
    for i, name in enumerate(target_names):
        diff_i = diff[:, :, :, i]
        mask_i = Y_mask_all[:, :, :, i]
        y_i = Y_all[:, :, :, i]
        
        mae_i = torch.sum(torch.abs(diff_i)) / (torch.sum(mask_i) + 1e-8)
        rmse_i = torch.sqrt(torch.sum(diff_i**2) / (torch.sum(mask_i) + 1e-8))
        
        y_mean_i = torch.sum(y_i * mask_i) / (torch.sum(mask_i) + 1e-8)
        ss_res_i = torch.sum((diff_i)**2)
        ss_tot_i = torch.sum(((y_i - y_mean_i) * mask_i)**2)
        r2_i = 1 - (ss_res_i / (ss_tot_i + 1e-8))
        
        print(f"\n{name} TEST results (REAL units):")
        print(f"{name}_MAE : {mae_i.item():.4f}")
        print(f"{name}_RMSE: {rmse_i.item():.4f}")
        print(f"{name}_R2  : {r2_i.item():.4f}")


if __name__ == "__main__":
    main()
