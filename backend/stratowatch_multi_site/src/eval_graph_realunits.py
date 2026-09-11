from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.graph_st_transformer import (
    StaticGraphSTTransformer,
    DynamicWindGraphSTTransformer,
)


SEED = 42
BATCH_SIZE = 64

D_MODEL = 128
NHEAD = 4
NUM_LAYERS_TIME = 2
NUM_LAYERS_SPACE = 2
DROPOUT = 0.1

GRAPH_BETA = 0.5
WIND_U_IDX = 8
WIND_V_IDX = 9


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inverse_scale(y, mean, std):
    return y * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def calculate_metrics(y_pred, y_true, mask):
    mask = mask.bool()

    diff = y_pred - y_true

    valid_diff = diff[mask]
    valid_true = y_true[mask]

    mae = torch.mean(torch.abs(valid_diff))
    rmse = torch.sqrt(torch.mean(valid_diff ** 2))

    y_mean = torch.mean(valid_true)
    ss_res = torch.sum(valid_diff ** 2)
    ss_tot = torch.sum((valid_true - y_mean) ** 2)

    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return mae.item(), rmse.item(), r2.item()


def calculate_target_metrics(y_pred, y_true, mask, target_idx):
    return calculate_metrics(
        y_pred[:, :, :, target_idx],
        y_true[:, :, :, target_idx],
        mask[:, :, :, target_idx],
    )


def main():

    parser = argparse.ArgumentParser(
        description="Final real-unit test evaluation for StratoWatch Graph-ST models."
    )

    parser.add_argument(
        "--model",
        choices=["static", "dynamic"],
        required=True,
        help="Graph-ST model to evaluate.",
    )

    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    project_root = os.path.dirname(os.path.dirname(__file__))

    npz_path = os.path.join(
        project_root,
        "data",
        "processed",
        "splits_final_Yscaled_Tin24_Tout6_stride1.npz",
    )

    if args.model == "static":
        model_name = "StaticGraphSTTransformer"
        checkpoint_name = "static_graph_st_best.pt"
    else:
        model_name = "DynamicWindGraphSTTransformer"
        checkpoint_name = "dynamic_wind_graph_st_best.pt"

    checkpoint_path = os.path.join(
        project_root,
        "outputs",
        "checkpoints",
        checkpoint_name,
    )

    scaler_path = os.path.join(
        project_root,
        "configs",
        "target_scaler.json",
    )

    adjacency_path = os.path.join(
        project_root,
        "data",
        "processed",
        "adjacency_final.npy",
    )

    coords_path = os.path.join(
        project_root,
        "configs",
        "site_coords_raw.npy",
    )

    output_dir = os.path.join(
        project_root,
        "outputs",
        "final_evaluation",
    )

    os.makedirs(output_dir, exist_ok=True)

    device = get_device()

    print("=" * 70)
    print("STRATOWATCH 2.0 — FINAL GRAPH-ST TEST EVALUATION")
    print("=" * 70)
    print("Device:", device)
    print("Seed:", SEED)
    print("Model:", model_name)
    print("Checkpoint:", checkpoint_path)
    print("Dataset:", npz_path)

    # --------------------------------------------------------
    # LOAD TARGET SCALER
    # --------------------------------------------------------

    with open(scaler_path, "r") as f:
        scaler = json.load(f)

    mean = torch.tensor(
        scaler["mean"],
        dtype=torch.float32,
        device=device,
    )

    std = torch.tensor(
        scaler["std"],
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    test_ds = MultisiteDataset(
        npz_path,
        split="test",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    num_features = test_ds.X.shape[-1]
    num_sites = test_ds.X.shape[2]
    tin = test_ds.tin
    tout = test_ds.tout

    print()
    print("TEST DATA")
    print("X:", test_ds.X.shape)
    print("Y:", test_ds.Y.shape)
    print("Features:", num_features)
    print("Sites:", num_sites)
    print("Tin:", tin)
    print("Tout:", tout)

    # --------------------------------------------------------
    # LOAD GRAPH DATA
    # --------------------------------------------------------

    adjacency = torch.tensor(
        np.load(adjacency_path),
        dtype=torch.float32,
        device=device,
    )

    site_coords = torch.tensor(
        np.load(coords_path),
        dtype=torch.float32,
        device=device,
    )

    print("Adjacency:", tuple(adjacency.shape))
    print("Coordinates:", tuple(site_coords.shape))

    # --------------------------------------------------------
    # BUILD EXACT TRAINING ARCHITECTURE
    # --------------------------------------------------------

    if args.model == "static":

        model = StaticGraphSTTransformer(
            num_features=num_features,
            num_targets=2,
            tin=tin,
            tout=tout,
            num_sites=num_sites,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers_time=NUM_LAYERS_TIME,
            num_layers_space=NUM_LAYERS_SPACE,
            dropout=DROPOUT,
        ).to(device)

    else:

        model = DynamicWindGraphSTTransformer(
            num_features=num_features,
            num_targets=2,
            tin=tin,
            tout=tout,
            num_sites=num_sites,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers_time=NUM_LAYERS_TIME,
            num_layers_space=NUM_LAYERS_SPACE,
            dropout=DROPOUT,
            graph_beta=GRAPH_BETA,
            wind_u_idx=WIND_U_IDX,
            wind_v_idx=WIND_V_IDX,
        ).to(device)

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state"],
    )

    model.eval()

    print("✓ Checkpoint loaded successfully")

    # --------------------------------------------------------
    # TEST INFERENCE
    # --------------------------------------------------------

    predictions = []
    truths = []
    masks = []

    print()
    print("Running test inference...")

    with torch.no_grad():

        for X, Y, X_mask, Y_mask in test_loader:

            X = X.to(device)
            Y = Y.to(device)
            Y_mask = Y_mask.to(device)

            output = model(
                X,
                adjacency,
                site_coords,
            )

            if isinstance(output, tuple):
                Y_hat = output[0]
            else:
                Y_hat = output

            predictions.append(
                Y_hat.cpu()
            )

            truths.append(
                Y.cpu()
            )

            masks.append(
                Y_mask.cpu()
            )

    Y_hat_scaled = torch.cat(
        predictions,
        dim=0,
    )

    Y_scaled = torch.cat(
        truths,
        dim=0,
    )

    Y_mask = torch.cat(
        masks,
        dim=0,
    ).bool()

    print("Predictions:", Y_hat_scaled.shape)
    print("Truth:", Y_scaled.shape)
    print("Mask:", Y_mask.shape)

    # --------------------------------------------------------
    # INVERSE TRANSFORM TO REAL UNITS
    # --------------------------------------------------------

    Y_hat_real = inverse_scale(
        Y_hat_scaled.to(device),
        mean,
        std,
    ).cpu()

    Y_real = inverse_scale(
        Y_scaled.to(device),
        mean,
        std,
    ).cpu()

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    overall_mae, overall_rmse, overall_r2 = calculate_metrics(
        Y_hat_real,
        Y_real,
        Y_mask,
    )

    o3_mae, o3_rmse, o3_r2 = calculate_target_metrics(
        Y_hat_real,
        Y_real,
        Y_mask,
        0,
    )

    no2_mae, no2_rmse, no2_r2 = calculate_target_metrics(
        Y_hat_real,
        Y_real,
        Y_mask,
        1,
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS — REAL UNITS")
    print("=" * 70)

    print()
    print("OVERALL")
    print(f"MAE : {overall_mae:.6f}")
    print(f"RMSE: {overall_rmse:.6f}")
    print(f"R²  : {overall_r2:.6f}")

    print()
    print("O3")
    print(f"MAE : {o3_mae:.6f}")
    print(f"RMSE: {o3_rmse:.6f}")
    print(f"R²  : {o3_r2:.6f}")

    print()
    print("NO2")
    print(f"MAE : {no2_mae:.6f}")
    print(f"RMSE: {no2_rmse:.6f}")
    print(f"R²  : {no2_r2:.6f}")

    # --------------------------------------------------------
    # HORIZON-WISE METRICS
    # --------------------------------------------------------

    horizon_results = []

    print()
    print("=" * 70)
    print("HORIZON-WISE RESULTS")
    print("=" * 70)

    for h in range(tout):

        pred_h = Y_hat_real[:, h:h + 1]
        true_h = Y_real[:, h:h + 1]
        mask_h = Y_mask[:, h:h + 1]

        mae_h, rmse_h, r2_h = calculate_metrics(
            pred_h,
            true_h,
            mask_h,
        )

        horizon_results.append(
            {
                "horizon": h + 1,
                "mae": mae_h,
                "rmse": rmse_h,
                "r2": r2_h,
            }
        )

        print(
            f"H+{h + 1} | "
            f"MAE={mae_h:.6f} | "
            f"RMSE={rmse_h:.6f} | "
            f"R²={r2_h:.6f}"
        )

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------

    np.save(
        os.path.join(
            output_dir,
            f"{args.model}_test_predictions_real.npy",
        ),
        Y_hat_real.numpy(),
    )

    np.save(
        os.path.join(
            output_dir,
            f"{args.model}_test_truth_real.npy",
        ),
        Y_real.numpy(),
    )

    np.save(
        os.path.join(
            output_dir,
            f"{args.model}_test_mask.npy",
        ),
        Y_mask.numpy(),
    )

    results = {
        "model": model_name,
        "checkpoint": checkpoint_path,
        "dataset": npz_path,
        "seed": SEED,
        "device": str(device),
        "num_features": int(num_features),
        "num_sites": int(num_sites),
        "tin": int(tin),
        "tout": int(tout),
        "overall": {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "r2": overall_r2,
        },
        "O3": {
            "mae": o3_mae,
            "rmse": o3_rmse,
            "r2": o3_r2,
        },
        "NO2": {
            "mae": no2_mae,
            "rmse": no2_rmse,
            "r2": no2_r2,
        },
        "horizon_wise": horizon_results,
    }

    results_path = os.path.join(
        output_dir,
        f"{args.model}_test_metrics_realunits.json",
    )

    with open(results_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("Metrics:", results_path)
    print("Predictions:", os.path.join(
        output_dir,
        f"{args.model}_test_predictions_real.npy",
    ))
    print("=" * 70)


if __name__ == "__main__":
    main()