"""
StratoWatch — Final Single-Site LSTM Baseline

Research contract:
    Input:
        24 hours × 134 features

    Output:
        6 forecast hours × 2 targets
        O3 + NO2

    Dataset:
        Official Phase 7 artifact

    Evaluation:
        Real-unit masked MAE
        Real-unit masked RMSE
        Real-unit masked R²
        Per-target metrics
        Horizon-wise metrics

This is a DIRECT-TARGET model.
It does NOT use the old residual-learning pipeline.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# IMPORT SHARED DATA LOADER
# ============================================================

try:
    from common_data import load_splits, inverse_transform_targets
except ImportError:
    from baselines.common_data import load_splits, inverse_transform_targets


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SINGLE_SITE_DIR = BASE_DIR.parent

OUTPUT_DIR = (
    SINGLE_SITE_DIR
    / "outputs"
    / "final_baselines"
    / "lstm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RESEARCH CONTRACT
# ============================================================

TIN = 24
NUM_FEATURES = 134
TOUT = 6
NUM_TARGETS = 2

TARGET_NAMES = [
    "O3",
    "NO2",
]


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

SEED = 42

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20

BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
PATIENCE = 5

GRADIENT_CLIP = 1.0


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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
# MODEL
# ============================================================

class LSTMForecaster(nn.Module):
    """
    Direct multi-horizon LSTM forecaster.

    Input:
        (B, 24, 134)

    Output:
        (B, 6, 2)
    """

    def __init__(
        self,
        input_size: int = NUM_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        output_size: int = NUM_TARGETS,
        forecast_horizon: int = TOUT,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                forecast_horizon * output_size,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output, _ = self.lstm(x)

        # Representation of the latest observed timestep.
        latest = output[:, -1, :]

        prediction = self.head(latest)

        return prediction.reshape(
            x.shape[0],
            self.forecast_horizon,
            self.output_size,
        )


# ============================================================
# MASKED LOSS
# ============================================================

def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    mask = mask.float()

    squared_error = (
        prediction - target
    ) ** 2

    weighted_error = (
        squared_error * mask
    )

    denominator = (
        mask.sum() + 1e-8
    )

    return (
        weighted_error.sum()
        / denominator
    )


# ============================================================
# EVALUATION LOSS
# ============================================================

def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:

    model.eval()

    total_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():

        for xb, yb, mb in loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            prediction = model(xb)

            mask = mb.float()

            squared_error = (
                prediction - yb
            ) ** 2

            total_loss += float(
                (
                    squared_error * mask
                ).sum().item()
            )

            total_weight += float(
                mask.sum().item()
            )

    if total_weight == 0:
        return float("nan")

    return total_loss / total_weight


# ============================================================
# METRIC HELPERS
# ============================================================

def masked_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:

    valid = mask.astype(bool)

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                y_pred[valid]
                - y_true[valid]
            )
        )
    )


def masked_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:

    valid = mask.astype(bool)

    if not np.any(valid):
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                (
                    y_pred[valid]
                    - y_true[valid]
                ) ** 2
            )
        )
    )


def masked_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:

    valid = mask.astype(bool)

    if not np.any(valid):
        return float("nan")

    true_values = y_true[valid]
    pred_values = y_pred[valid]

    mean_true = np.mean(true_values)

    sse = np.sum(
        (
            true_values
            - pred_values
        ) ** 2
    )

    sst = np.sum(
        (
            true_values
            - mean_true
        ) ** 2
    )

    if sst <= 0:
        return float("nan")

    return float(
        1.0 - sse / sst
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> Dict:

    results: Dict = {}

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    results["overall"] = {
        "MAE": masked_mae(
            y_true,
            y_pred,
            mask,
        ),
        "RMSE": masked_rmse(
            y_true,
            y_pred,
            mask,
        ),
        "R2": masked_r2(
            y_true,
            y_pred,
            mask,
        ),
    }

    # --------------------------------------------------------
    # Target-wise
    # --------------------------------------------------------

    results["targets"] = {}

    for target_idx, target_name in enumerate(
        TARGET_NAMES
    ):

        true_target = y_true[
            :,
            :,
            target_idx,
        ]

        pred_target = y_pred[
            :,
            :,
            target_idx,
        ]

        mask_target = mask[
            :,
            :,
            target_idx,
        ]

        results["targets"][
            target_name
        ] = {
            "MAE": masked_mae(
                true_target,
                pred_target,
                mask_target,
            ),
            "RMSE": masked_rmse(
                true_target,
                pred_target,
                mask_target,
            ),
            "R2": masked_r2(
                true_target,
                pred_target,
                mask_target,
            ),
        }

    # --------------------------------------------------------
    # Horizon-wise
    # --------------------------------------------------------

    results["horizons"] = {}

    for h in range(
        y_true.shape[1]
    ):

        true_h = y_true[:, h, :]
        pred_h = y_pred[:, h, :]
        mask_h = mask[:, h, :]

        results["horizons"][
            f"H+{h + 1}"
        ] = {
            "MAE": masked_mae(
                true_h,
                pred_h,
                mask_h,
            ),
            "RMSE": masked_rmse(
                true_h,
                pred_h,
                mask_h,
            ),
            "R2": masked_r2(
                true_h,
                pred_h,
                mask_h,
            ),
        }

    return results


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:

    model.eval()

    predictions = []

    with torch.no_grad():

        for xb, _, _ in loader:

            xb = xb.to(device)

            prediction = model(
                xb
            ).cpu().numpy()

            predictions.append(
                prediction
            )

    return np.concatenate(
        predictions,
        axis=0,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_seed(SEED)

    print()
    print("=" * 70)
    print("STRATOWATCH — FINAL SINGLE-SITE LSTM")
    print("=" * 70)

    total_start = time.time()

    device = get_device()

    print()
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("[1/7] Loading official Phase 7 dataset...")

    splits = load_splits()

    X_train = splits["X_train"]
    Y_train = splits["Y_train"]

    X_val = splits["X_val"]
    Y_val = splits["Y_val"]

    X_test = splits["X_test"]
    Y_test = splits["Y_test"]

    Y_mask_train = splits[
        "Y_mask_train"
    ]

    Y_mask_val = splits[
        "Y_mask_val"
    ]

    Y_mask_test = splits[
        "Y_mask_test"
    ]

    target_scaler = splits[
        "target_scaler"
    ]

    # --------------------------------------------------------
    # Contract checks
    # --------------------------------------------------------

    expected_x_train = (
        X_train.shape[0],
        TIN,
        NUM_FEATURES,
    )

    expected_y_train = (
        Y_train.shape[0],
        TOUT,
        NUM_TARGETS,
    )

    if X_train.shape != expected_x_train:
        raise ValueError(
            f"Unexpected X_train shape: "
            f"{X_train.shape}"
        )

    if Y_train.shape != expected_y_train:
        raise ValueError(
            f"Unexpected Y_train shape: "
            f"{Y_train.shape}"
        )

    print()
    print("Dataset:")
    print(
        f"  X_train: {X_train.shape}"
    )
    print(
        f"  Y_train: {Y_train.shape}"
    )
    print(
        f"  X_val  : {X_val.shape}"
    )
    print(
        f"  Y_val  : {Y_val.shape}"
    )
    print(
        f"  X_test : {X_test.shape}"
    )
    print(
        f"  Y_test : {Y_test.shape}"
    )

    # ========================================================
    # CREATE TENSOR DATASETS
    # ========================================================

    print()
    print("[2/7] Creating PyTorch datasets...")

    X_train_t = torch.from_numpy(
        X_train.astype(np.float32)
    )

    Y_train_t = torch.from_numpy(
        Y_train.astype(np.float32)
    )

    M_train_t = torch.from_numpy(
        Y_mask_train.astype(np.float32)
    )

    X_val_t = torch.from_numpy(
        X_val.astype(np.float32)
    )

    Y_val_t = torch.from_numpy(
        Y_val.astype(np.float32)
    )

    M_val_t = torch.from_numpy(
        Y_mask_val.astype(np.float32)
    )

    X_test_t = torch.from_numpy(
        X_test.astype(np.float32)
    )

    Y_test_t = torch.from_numpy(
        Y_test.astype(np.float32)
    )

    M_test_t = torch.from_numpy(
        Y_mask_test.astype(np.float32)
    )

    train_dataset = TensorDataset(
        X_train_t,
        Y_train_t,
        M_train_t,
    )

    val_dataset = TensorDataset(
        X_val_t,
        Y_val_t,
        M_val_t,
    )

    test_dataset = TensorDataset(
        X_test_t,
        Y_test_t,
        M_test_t,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print()
    print("[3/7] Building LSTM...")

    model = LSTMForecaster(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=NUM_TARGETS,
        forecast_horizon=TOUT,
        dropout=DROPOUT,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("LSTM configuration:")
    print(
        f"  Input features : {NUM_FEATURES}"
    )
    print(
        f"  Input hours    : {TIN}"
    )
    print(
        f"  Hidden size    : {HIDDEN_SIZE}"
    )
    print(
        f"  LSTM layers    : {NUM_LAYERS}"
    )
    print(
        f"  Dropout        : {DROPOUT}"
    )
    print(
        f"  Output horizon : {TOUT}"
    )
    print(
        f"  Targets        : {NUM_TARGETS}"
    )
    print(
        f"  Parameters     : {parameter_count:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("[4/7] Training LSTM...")
    print()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_model_path = (
        OUTPUT_DIR
        / "lstm_best.pt"
    )

    train_start = time.time()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_weight = 0.0

        for xb, yb, mb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(xb)

            loss = masked_mse_loss(
                prediction,
                yb,
                mb,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss "
                    f"at epoch {epoch}: {loss.item()}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            optimizer.step()

            weight = float(
                mb.sum().item()
            )

            total_loss += (
                float(loss.item())
                * weight
            )

            total_weight += weight

        train_loss = (
            total_loss
            / max(total_weight, 1.0)
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} "
            f"| train_MSE={train_loss:.6f} "
            f"| val_MSE={val_loss:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "model_config": {
                        "input_size":
                            NUM_FEATURES,
                        "hidden_size":
                            HIDDEN_SIZE,
                        "num_layers":
                            NUM_LAYERS,
                        "output_size":
                            NUM_TARGETS,
                        "forecast_horizon":
                            TOUT,
                        "dropout":
                            DROPOUT,
                    },
                    "research_contract": {
                        "features":
                            NUM_FEATURES,
                        "tin":
                            TIN,
                        "tout":
                            TOUT,
                        "targets":
                            TARGET_NAMES,
                    },
                    "seed": SEED,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_model_path,
            )

            print(
                "  ✓ best checkpoint saved"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"Early stopping after "
                    f"{epoch} epochs"
                )

                break

    train_seconds = (
        time.time() - train_start
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print("[5/7] Loading best checkpoint...")

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"  Best epoch: {best_epoch}"
    )

    print(
        f"  Best validation MSE: "
        f"{best_val_loss:.6f}"
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print()
    print("[6/7] Generating predictions...")

    val_pred_scaled = predict(
        model,
        val_loader,
        device,
    )

    test_pred_scaled = predict(
        model,
        test_loader,
        device,
    )

    print(
        f"  Validation predictions: "
        f"{val_pred_scaled.shape}"
    )

    print(
        f"  Test predictions: "
        f"{test_pred_scaled.shape}"
    )

    # ========================================================
    # INVERSE TRANSFORM
    # ========================================================

    print()
    print(
        "Converting predictions "
        "to real units..."
    )

    Y_val_real = inverse_transform_targets(
        Y_val,
        target_scaler,
    )

    Y_test_real = inverse_transform_targets(
        Y_test,
        target_scaler,
    )

    val_pred_real = inverse_transform_targets(
        val_pred_scaled,
        target_scaler,
    )

    test_pred_real = inverse_transform_targets(
        test_pred_scaled,
        target_scaler,
    )

    # ========================================================
    # METRICS
    # ========================================================

    print()
    print("[7/7] Calculating masked real-unit metrics...")

    val_metrics = calculate_metrics(
        Y_val_real,
        val_pred_real,
        Y_mask_val,
    )

    test_metrics = calculate_metrics(
        Y_test_real,
        test_pred_real,
        Y_mask_test,
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("LSTM — FINAL TEST RESULTS")
    print("=" * 70)

    print()
    print("OVERALL")

    print(
        f"MAE : "
        f"{test_metrics['overall']['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{test_metrics['overall']['RMSE']:.4f}"
    )

    print(
        f"R²  : "
        f"{test_metrics['overall']['R2']:.4f}"
    )

    print()
    print("O3")

    print(
        f"MAE : "
        f"{test_metrics['targets']['O3']['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{test_metrics['targets']['O3']['RMSE']:.4f}"
    )

    print(
        f"R²  : "
        f"{test_metrics['targets']['O3']['R2']:.4f}"
    )

    print()
    print("NO2")

    print(
        f"MAE : "
        f"{test_metrics['targets']['NO2']['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{test_metrics['targets']['NO2']['RMSE']:.4f}"
    )

    print(
        f"R²  : "
        f"{test_metrics['targets']['NO2']['R2']:.4f}"
    )

    print()
    print("HORIZON-WISE")

    for horizon, metrics in test_metrics[
        "horizons"
    ].items():

        print(
            f"{horizon} | "
            f"MAE={metrics['MAE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"R²={metrics['R2']:.4f}"
        )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    np.save(
        OUTPUT_DIR / "test_predictions.npy",
        test_pred_real,
    )

    np.save(
        OUTPUT_DIR / "test_truth.npy",
        Y_test_real,
    )

    np.save(
        OUTPUT_DIR / "test_mask.npy",
        Y_mask_test,
    )

    np.save(
        OUTPUT_DIR / "val_predictions.npy",
        val_pred_real,
    )

    np.save(
        OUTPUT_DIR / "val_truth.npy",
        Y_val_real,
    )

    np.save(
        OUTPUT_DIR / "val_mask.npy",
        Y_mask_val,
    )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_output = {
        "model": "lstm",
        "model_type": "LSTMForecaster",
        "target_mode":
            "direct_target_prediction",
        "dataset": {
            "artifact":
                splits["data_path"],
            "features":
                NUM_FEATURES,
            "tin":
                TIN,
            "tout":
                TOUT,
            "targets":
                TARGET_NAMES,
            "train_samples":
                int(X_train.shape[0]),
            "val_samples":
                int(X_val.shape[0]),
            "test_samples":
                int(X_test.shape[0]),
        },
        "configuration": {
            "hidden_size":
                HIDDEN_SIZE,
            "num_layers":
                NUM_LAYERS,
            "dropout":
                DROPOUT,
            "batch_size":
                BATCH_SIZE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "max_epochs":
                MAX_EPOCHS,
            "patience":
                PATIENCE,
            "gradient_clip":
                GRADIENT_CLIP,
            "seed":
                SEED,
        },
        "best_epoch":
            best_epoch,
        "best_validation_mse":
            best_val_loss,
        "validation":
            val_metrics,
        "test":
            test_metrics,
        "training_seconds":
            train_seconds,
        "total_runtime_seconds":
            time.time() - total_start,
    }

    metrics_path = (
        OUTPUT_DIR
        / "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_output,
            f,
            indent=2,
        )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "model":
            "lstm",
        "model_type":
            "LSTMForecaster",
        "direct_target_prediction":
            True,
        "residual_learning":
            False,
        "dataset_artifact":
            splits["data_path"],
        "feature_schema":
            splits["feature_list_path"],
        "target_scaler":
            splits["target_scaler_path"],
        "contract": {
            "feature_count":
                NUM_FEATURES,
            "input_hours":
                TIN,
            "forecast_hours":
                TOUT,
            "target_count":
                NUM_TARGETS,
            "targets":
                TARGET_NAMES,
        },
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("LSTM COMPLETE")
    print("=" * 70)

    print()
    print("Saved:")
    print(
        f"  Checkpoint : "
        f"{best_model_path}"
    )

    print(
        f"  Metrics    : "
        f"{metrics_path}"
    )

    print(
        f"  Predictions: "
        f"{OUTPUT_DIR / 'test_predictions.npy'}"
    )

    print(
        f"  History    : "
        f"{OUTPUT_DIR / 'training_history.json'}"
    )

    print(
        f"  Metadata   : "
        f"{OUTPUT_DIR / 'metadata.json'}"
    )

    print()
    print(
        f"Total runtime: "
        f"{time.time() - total_start:.2f} seconds"
    )

    print()
    print("RESEARCH CONTRACT: PASSED")
    print("LSTM: COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__": 
    main()