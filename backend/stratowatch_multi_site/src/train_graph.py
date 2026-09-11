
"""
src/train_graph.py

Final multi-site Graph-ST Transformer training script for StratoWatch 2.0.

Models:
    --model static
        StaticGraphSTTransformer

    --model dynamic
        DynamicWindGraphSTTransformer

Uses:
    Phase 7 final Y-scaled multi-site artifact.

Input:
    (B, Tin, S, F)

Output:
    (B, Tout, S, 2)

Targets:
    O3, NO2

Both models perform DIRECT target prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.graph_st_transformer import (
    StaticGraphSTTransformer,
    DynamicWindGraphSTTransformer,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 32

LR = 1e-4
WEIGHT_DECAY = 1e-4

# Controlled short run to finish Phase 8 efficiently.
EPOCHS = 5
PATIENCE = 5

D_MODEL = 128
NHEAD = 4

NUM_LAYERS_TIME = 2
NUM_LAYERS_SPACE = 2

DROPOUT = 0.1

# Dynamic graph configuration
GRAPH_BETA = 0.5
WIND_U_IDX = 8
WIND_V_IDX = 9

# CPU is intentionally used.
# The previous MPS run produced invalid inf losses.
USE_MPS = False


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    if USE_MPS and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# MASKED HUBER LOSS
# ============================================================

def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:

    diff = pred - target
    abs_diff = diff.abs()

    quadratic = torch.minimum(
        abs_diff,
        torch.tensor(
            delta,
            device=diff.device,
            dtype=diff.dtype,
        ),
    )

    linear = abs_diff - quadratic

    loss = (
        0.5 * quadratic.square()
        + delta * linear
    )

    mask = mask.to(
        dtype=loss.dtype
    )

    return (
        loss * mask
    ).sum() / mask.sum().clamp_min(1.0)


# ============================================================
# GLOBAL MASKED METRICS
# ============================================================

def masked_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float, float]:

    mask = mask.to(
        dtype=pred.dtype
    )

    diff = pred - target

    abs_sum = float(
        (diff.abs() * mask)
        .sum()
        .item()
    )

    sq_sum = float(
        (diff.square() * mask)
        .sum()
        .item()
    )

    count = float(
        mask.sum().item()
    )

    return (
        abs_sum,
        sq_sum,
        count,
    )


def evaluate(
    model,
    loader,
    device: torch.device,
    adjacency: torch.Tensor,
    site_coords: torch.Tensor,
):
    model.eval()

    abs_sum = 0.0
    sq_sum = 0.0
    count = 0.0

    with torch.no_grad():

        for X, Y, X_mask, Y_mask in loader:

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

            a, s, n = masked_sums(
                Y_hat,
                Y,
                Y_mask,
            )

            abs_sum += a
            sq_sum += s
            count += n

    mae = (
        abs_sum
        / max(count, 1.0)
    )

    rmse = float(
        np.sqrt(
            sq_sum
            / max(count, 1.0)
        )
    )

    return (
        mae,
        rmse,
        count,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train StratoWatch 2.0 "
            "Graph-ST Transformer."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "static",
            "dynamic",
        ],
        required=True,
        help=(
            "Graph model variant to train."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    device = get_device()

    # --------------------------------------------------------
    # Project paths
    # --------------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(__file__)
    )

    npz_path = os.path.join(
        project_root,
        "data",
        "processed",
        "splits_final_Yscaled_Tin24_Tout6_stride1.npz",
    )

    adjacency_path = os.path.join(
        project_root,
        "data",
        "processed",
        "adjacency_final.npy",
    )

    coordinates_path = os.path.join(
        project_root,
        "configs",
        "site_coords_raw.npy",
    )

    out_dir = os.path.join(
        project_root,
        "outputs",
        "checkpoints",
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    if args.model == "static":

        model_label = (
            "StaticGraphSTTransformer"
        )

        model_type = (
            "StaticGraphSTTransformer"
        )

        best_filename = (
            "static_graph_st_best.pt"
        )

        history_filename = (
            "static_graph_st_training_history.json"
        )

    else:

        model_label = (
            "DynamicWindGraphSTTransformer"
        )

        model_type = (
            "DynamicWindGraphSTTransformer"
        )

        best_filename = (
            "dynamic_wind_graph_st_best.pt"
        )

        history_filename = (
            "dynamic_wind_graph_st_training_history.json"
        )

    best_path = os.path.join(
        out_dir,
        best_filename,
    )

    history_path = os.path.join(
        out_dir,
        history_filename,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 70)

    print(
        f"STRATOWATCH 2.0 — FINAL {model_label}"
    )

    print("=" * 70)

    print(
        "Device:",
        device,
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "Model:",
        model_label,
    )

    print(
        "Dataset:",
        npz_path,
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_ds = MultisiteDataset(
        npz_path,
        split="train",
    )

    val_ds = MultisiteDataset(
        npz_path,
        split="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    num_features = (
        train_ds.X.shape[-1]
    )

    num_sites = (
        train_ds.X.shape[2]
    )

    tin = train_ds.tin
    tout = train_ds.tout

    print()

    print(
        "Train X:",
        train_ds.X.shape,
    )

    print(
        "Train Y:",
        train_ds.Y.shape,
    )

    print(
        "Val X  :",
        val_ds.X.shape,
    )

    print(
        "Val Y  :",
        val_ds.Y.shape,
    )

    print(
        "Features:",
        num_features,
    )

    print(
        "Sites:",
        num_sites,
    )

    print(
        "Tin:",
        tin,
    )

    print(
        "Tout:",
        tout,
    )

    # --------------------------------------------------------
    # GRAPH DATA
    # --------------------------------------------------------

    adjacency_np = np.load(
        adjacency_path
    )

    site_coords_np = np.load(
        coordinates_path
    )

    adjacency = torch.tensor(
        adjacency_np,
        dtype=torch.float32,
        device=device,
    )

    site_coords = torch.tensor(
        site_coords_np,
        dtype=torch.float32,
        device=device,
    )

    print(
        "Adjacency:",
        adjacency_np.shape,
    )

    print(
        "Coordinates:",
        site_coords_np.shape,
    )

    # --------------------------------------------------------
    # MODEL
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
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    best_val_mae = float(
        "inf"
    )

    best_epoch = 0

    bad_epochs = 0

    history = []

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print("TRAINING")

    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        start_time = time.time()

        model.train()

        train_loss_sum = 0.0

        train_valid_count = 0.0

        for X, Y, X_mask, Y_mask in train_loader:

            X = X.to(device)
            Y = Y.to(device)
            Y_mask = Y_mask.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                X,
                adjacency,
                site_coords,
            )

            if isinstance(output, tuple):
                Y_hat = output[0]
            else:
                Y_hat = output

            loss = masked_huber(
                Y_hat,
                Y,
                Y_mask,
                delta=1.0,
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite training loss "
                    "detected. Training stopped."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            valid = float(
                Y_mask.sum().item()
            )

            train_loss_sum += (
                float(loss.item())
                * valid
            )

            train_valid_count += valid

        train_huber = (
            train_loss_sum
            / max(
                train_valid_count,
                1.0,
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_mae, val_rmse, val_count = evaluate(
            model,
            val_loader,
            device,
            adjacency,
            site_coords,
        )

        elapsed = (
            time.time()
            - start_time
        )

        record = {
            "epoch": epoch,
            "train_huber": train_huber,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_valid_count": val_count,
            "elapsed_seconds": elapsed,
        }

        history.append(
            record
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_Huber={train_huber:.4f} | "
            f"val_MAE={val_mae:.4f} | "
            f"val_RMSE={val_rmse:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if (
            np.isfinite(val_mae)
            and val_mae < best_val_mae
        ):

            best_val_mae = val_mae

            best_epoch = epoch

            bad_epochs = 0

            checkpoint = {
                "model_state": model.state_dict(),

                "model_type": model_type,

                "architecture": {
                    "num_features": num_features,
                    "num_targets": 2,
                    "num_sites": num_sites,
                    "tin": tin,
                    "tout": tout,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers_time": (
                        NUM_LAYERS_TIME
                    ),
                    "num_layers_space": (
                        NUM_LAYERS_SPACE
                    ),
                    "dropout": DROPOUT,
                },

                "training": {
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "epochs_requested": EPOCHS,
                    "patience": PATIENCE,
                    "device": str(device),
                },

                "graph": {
                    "adjacency_path": adjacency_path,
                    "coordinates_path": coordinates_path,
                    "graph_beta": (
                        GRAPH_BETA
                        if args.model == "dynamic"
                        else None
                    ),
                    "wind_u_idx": (
                        WIND_U_IDX
                        if args.model == "dynamic"
                        else None
                    ),
                    "wind_v_idx": (
                        WIND_V_IDX
                        if args.model == "dynamic"
                        else None
                    ),
                },

                "epoch": epoch,

                "val_mae": val_mae,

                "val_rmse": val_rmse,

                "feature_cols": (
                    train_ds.feature_cols
                ),

                "target_cols": (
                    train_ds.target_cols
                ),

                "dataset": {
                    "path": npz_path,
                    "artifact": os.path.basename(
                        npz_path
                    ),
                    "phase": (
                        "Phase 7 final "
                        "multi-site Y-scaled artifact"
                    ),
                },

                "created_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                "Saved best checkpoint:",
                best_path,
            )

        else:

            bad_epochs += 1

            if bad_epochs >= PATIENCE:

                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no validation improvement for "
                    f"{PATIENCE} epochs)."
                )

                break

        # ----------------------------------------------------
        # SAVE HISTORY AFTER EVERY EPOCH
        # ----------------------------------------------------

        history_payload = {
            "model_type": model_type,
            "seed": SEED,
            "best_epoch": best_epoch,
            "best_val_mae": (
                float(best_val_mae)
                if np.isfinite(
                    best_val_mae
                )
                else None
            ),
            "dataset": os.path.basename(
                npz_path
            ),

            "architecture": {
                "num_features": num_features,
                "num_targets": 2,
                "num_sites": num_sites,
                "tin": tin,
                "tout": tout,
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers_time": (
                    NUM_LAYERS_TIME
                ),
                "num_layers_space": (
                    NUM_LAYERS_SPACE
                ),
                "dropout": DROPOUT,
            },

            "history": history,
        }

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history_payload,
                f,
                indent=2,
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "FINAL GRAPH-ST TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        "Model:",
        model_label,
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation MAE:",
        best_val_mae,
    )

    print(
        "Checkpoint:",
        best_path,
    )

    print(
        "History:",
        history_path,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

