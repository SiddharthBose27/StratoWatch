"""
src/train_baseline.py

Final multi-site ST Transformer training script for StratoWatch 2.0.

Uses the Phase 7 final Y-scaled multi-site artifact and a controlled
research-training configuration.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultisiteDataset
from src.models.st_transformer import STTransformer


SEED = 42

BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 5

D_MODEL = 128
NHEAD = 4
NUM_LAYERS_TIME = 2
NUM_LAYERS_SPACE = 2
DROPOUT = 0.1

# CPU is used for the controlled final run.
USE_MPS = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if USE_MPS and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        torch.tensor(delta, device=diff.device, dtype=diff.dtype),
    )

    linear = abs_diff - quadratic

    loss = 0.5 * quadratic.square() + delta * linear
    mask = mask.to(dtype=loss.dtype)

    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float, float]:

    mask = mask.to(dtype=pred.dtype)
    diff = pred - target

    abs_sum = float((diff.abs() * mask).sum().item())
    sq_sum = float((diff.square() * mask).sum().item())
    count = float(mask.sum().item())

    return abs_sum, sq_sum, count


def evaluate(
    model,
    loader,
    device: torch.device,
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

            output = model(X)

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

    mae = abs_sum / max(count, 1.0)
    rmse = float(np.sqrt(sq_sum / max(count, 1.0)))

    return mae, rmse, count


def main():

    set_seed(SEED)

    project_root = os.path.dirname(
        os.path.dirname(__file__)
    )

    npz_path = os.path.join(
        project_root,
        "data",
        "processed",
        "splits_final_Yscaled_Tin24_Tout6_stride1.npz",
    )

    out_dir = os.path.join(
        project_root,
        "outputs",
        "checkpoints",
    )

    os.makedirs(out_dir, exist_ok=True)

    best_path = os.path.join(
        out_dir,
        "st_transformer_best.pt",
    )

    history_path = os.path.join(
        out_dir,
        "st_transformer_training_history.json",
    )

    device = get_device()

    print("=" * 70)
    print("STRATOWATCH 2.0 — FINAL MULTI-SITE ST TRANSFORMER")
    print("=" * 70)

    print("Device:", device)
    print("Seed:", SEED)
    print("Dataset:", npz_path)

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

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

    num_features = train_ds.X.shape[-1]
    num_sites = train_ds.X.shape[2]
    tin = train_ds.tin
    tout = train_ds.tout

    print()
    print("Train X:", train_ds.X.shape)
    print("Train Y:", train_ds.Y.shape)
    print("Val X  :", val_ds.X.shape)
    print("Val Y  :", val_ds.Y.shape)
    print("Features:", num_features)
    print("Sites:", num_sites)
    print("Tin:", tin)
    print("Tout:", tout)

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = STTransformer(
        num_features=num_features,
        tin=tin,
        tout=tout,
        num_sites=num_sites,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers_time=NUM_LAYERS_TIME,
        num_layers_space=NUM_LAYERS_SPACE,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0

    history = []

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        start_time = time.time()

        model.train()

        train_loss_sum = 0.0
        train_valid_count = 0.0

        for X, Y, X_mask, Y_mask in train_loader:

            X = X.to(device)
            Y = Y.to(device)
            Y_mask = Y_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(X)

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

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            valid = float(
                Y_mask.sum().item()
            )

            train_loss_sum += (
                float(loss.item()) * valid
            )

            train_valid_count += valid

        train_huber = (
            train_loss_sum
            / max(train_valid_count, 1.0)
        )

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        val_mae, val_rmse, val_count = evaluate(
            model,
            val_loader,
            device,
        )

        elapsed = time.time() - start_time

        record = {
            "epoch": epoch,
            "train_huber": train_huber,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_valid_count": val_count,
            "elapsed_seconds": elapsed,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"train_Huber={train_huber:.4f} | "
            f"val_MAE={val_mae:.4f} | "
            f"val_RMSE={val_rmse:.4f} | "
            f"{elapsed:.1f}s"
        )

        # --------------------------------------------------------
        # BEST CHECKPOINT
        # --------------------------------------------------------

        if val_mae < best_val_mae:

            best_val_mae = val_mae
            best_epoch = epoch
            bad_epochs = 0

            checkpoint = {
                "model_state": model.state_dict(),

                "model_type": "STTransformer",

                "architecture": {
                    "num_features": num_features,
                    "num_sites": num_sites,
                    "tin": tin,
                    "tout": tout,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers_time": NUM_LAYERS_TIME,
                    "num_layers_space": NUM_LAYERS_SPACE,
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

                "epoch": epoch,

                "val_mae": val_mae,
                "val_rmse": val_rmse,

                "feature_cols": train_ds.feature_cols,
                "target_cols": train_ds.target_cols,

                "dataset": {
                    "path": npz_path,
                    "artifact": os.path.basename(npz_path),
                    "phase": (
                        "Phase 7 final multi-site "
                        "Y-scaled artifact"
                    ),
                },

                "created_at_utc": (
                    datetime.now(timezone.utc).isoformat()
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

    # ------------------------------------------------------------
    # SAVE HISTORY
    # ------------------------------------------------------------

    history_payload = {
        "model_type": "STTransformer",
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "dataset": os.path.basename(npz_path),

        "architecture": {
            "num_features": num_features,
            "num_sites": num_sites,
            "tin": tin,
            "tout": tout,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers_time": NUM_LAYERS_TIME,
            "num_layers_space": NUM_LAYERS_SPACE,
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

    print()
    print("=" * 70)
    print("FINAL ST TRANSFORMER TRAINING COMPLETE")
    print("=" * 70)
    print("Best epoch:", best_epoch)
    print("Best validation MAE:", best_val_mae)
    print("Checkpoint:", best_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()