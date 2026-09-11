"""
StratoWatch — Final Single-Site LSTM Evaluation

Evaluation-only script.

Loads the frozen LSTM checkpoint produced by train_lstm.py
and evaluates it on the official Phase 7 test artifact.

Research contract:
    Input:
        24 hours × 134 features

    Output:
        6 forecast hours × 2 targets
        O3 + NO2

    Evaluation:
        Real-unit masked MAE
        Real-unit masked RMSE
        Real-unit masked R²
        Per-target metrics
        Horizon-wise metrics

IMPORTANT:
    - NO RETRAINING
    - NO DATA MODIFICATION
    - NO ARCHITECTURE CHANGES
    - USES FROZEN CHECKPOINT ONLY
"""

from __future__ import annotations

import json
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
    from common_data import (
        load_splits,
        inverse_transform_targets,
    )
except ImportError:
    from baselines.common_data import (
        load_splits,
        inverse_transform_targets,
    )


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

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "lstm_best.pt"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "test_metrics.json"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "test_predictions.npy"
)

TRUTH_PATH = (
    OUTPUT_DIR
    / "test_truth.npy"
)

MASK_PATH = (
    OUTPUT_DIR
    / "test_mask.npy"
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

BATCH_SIZE = 256


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:

    # Evaluation can safely use CPU for reproducibility
    # and to avoid MPS checkpoint/device issues.
    return torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class LSTMForecaster(nn.Module):
    """
    Exact architecture used by train_lstm.py.

    Input:
        (B, 24, 134)

    Output:
        (B, 6, 2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        forecast_horizon: int,
        dropout: float,
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

        latest = output[:, -1, :]

        prediction = self.head(latest)

        return prediction.reshape(
            x.shape[0],
            self.forecast_horizon,
            self.output_size,
        )


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


# ============================================================
# METRICS
# ============================================================

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
    # Per target
    # --------------------------------------------------------

    results["per_target"] = {}

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

        results["per_target"][
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

    results["horizon_wise"] = {}

    for h in range(TOUT):

        true_h = y_true[:, h, :]
        pred_h = y_pred[:, h, :]
        mask_h = mask[:, h, :]

        results["horizon_wise"][
            f"horizon_{h + 1}"
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

        for (xb,) in loader:

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

    total_start = time.time()

    print()
    print("=" * 70)
    print("STRATOWATCH — LSTM FINAL EVALUATION")
    print("=" * 70)

    print()
    print("EVALUATION ONLY")
    print("RETRAINING: NO")
    print("MODEL: FROZEN")
    print("DATASET: OFFICIAL PHASE 7 TEST")
    print("EVALUATION: REAL-UNIT MASKED")

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print()
    print(f"Device: {device}")

    # ========================================================
    # CHECKPOINT
    # ========================================================

    print()
    print("[1/6] Loading frozen LSTM checkpoint...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Frozen checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unexpected checkpoint format."
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model_config = checkpoint.get(
        "model_config",
        {},
    )

    input_size = int(
        model_config.get(
            "input_size",
            NUM_FEATURES,
        )
    )

    hidden_size = int(
        model_config.get(
            "hidden_size",
            128,
        )
    )

    num_layers = int(
        model_config.get(
            "num_layers",
            2,
        )
    )

    output_size = int(
        model_config.get(
            "output_size",
            NUM_TARGETS,
        )
    )

    forecast_horizon = int(
        model_config.get(
            "forecast_horizon",
            TOUT,
        )
    )

    dropout = float(
        model_config.get(
            "dropout",
            0.20,
        )
    )

    # --------------------------------------------------------
    # Contract validation
    # --------------------------------------------------------

    if input_size != NUM_FEATURES:
        raise ValueError(
            f"Checkpoint input_size={input_size}, "
            f"expected {NUM_FEATURES}."
        )

    if output_size != NUM_TARGETS:
        raise ValueError(
            f"Checkpoint output_size={output_size}, "
            f"expected {NUM_TARGETS}."
        )

    if forecast_horizon != TOUT:
        raise ValueError(
            f"Checkpoint forecast_horizon="
            f"{forecast_horizon}, expected {TOUT}."
        )

    research_contract = checkpoint.get(
        "research_contract",
        {},
    )

    if research_contract:

        contract_features = research_contract.get(
            "features"
        )

        contract_tin = research_contract.get(
            "tin"
        )

        contract_tout = research_contract.get(
            "tout"
        )

        if contract_features is not None:
            if int(contract_features) != NUM_FEATURES:
                raise ValueError(
                    "Research contract feature count "
                    "does not match."
                )

        if contract_tin is not None:
            if int(contract_tin) != TIN:
                raise ValueError(
                    "Research contract TIN "
                    "does not match."
                )

        if contract_tout is not None:
            if int(contract_tout) != TOUT:
                raise ValueError(
                    "Research contract TOUT "
                    "does not match."
                )

    print(
        f"  Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"  Best epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"  Validation MSE: "
        f"{checkpoint.get('val_loss', 'unknown')}"
    )

    print()
    print("  Model configuration:")
    print(
        f"    Input size      : {input_size}"
    )
    print(
        f"    Hidden size     : {hidden_size}"
    )
    print(
        f"    LSTM layers     : {num_layers}"
    )
    print(
        f"    Output size     : {output_size}"
    )
    print(
        f"    Forecast horizon: {forecast_horizon}"
    )
    print(
        f"    Dropout         : {dropout}"
    )

    # ========================================================
    # LOAD OFFICIAL DATA
    # ========================================================

    print()
    print("[2/6] Loading official Phase 7 test data...")

    splits = load_splits()

    X_test = splits["X_test"]
    Y_test = splits["Y_test"]
    Y_mask_test = splits["Y_mask_test"]
    target_scaler = splits["target_scaler"]

    print(
        f"  X_test : {X_test.shape}"
    )

    print(
        f"  Y_test : {Y_test.shape}"
    )

    print(
        f"  Mask   : {Y_mask_test.shape}"
    )

    # --------------------------------------------------------
    # Exact shape checks
    # --------------------------------------------------------

    expected_x_shape = (
        X_test.shape[0],
        TIN,
        NUM_FEATURES,
    )

    expected_y_shape = (
        X_test.shape[0],
        TOUT,
        NUM_TARGETS,
    )

    if X_test.shape != expected_x_shape:
        raise ValueError(
            f"Unexpected X_test shape: "
            f"{X_test.shape}; "
            f"expected {expected_x_shape}"
        )

    if Y_test.shape != expected_y_shape:
        raise ValueError(
            f"Unexpected Y_test shape: "
            f"{Y_test.shape}; "
            f"expected {expected_y_shape}"
        )

    if Y_mask_test.shape != expected_y_shape:
        raise ValueError(
            f"Unexpected mask shape: "
            f"{Y_mask_test.shape}; "
            f"expected {expected_y_shape}"
        )

    if not np.all(
        np.isfinite(X_test)
    ):
        raise ValueError(
            "X_test contains non-finite values."
        )

    if not np.all(
        np.isfinite(Y_test)
    ):
        raise ValueError(
            "Y_test contains non-finite values."
        )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print()
    print("[3/6] Reconstructing exact LSTM architecture...")

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        forecast_horizon=forecast_horizon,
        dropout=dropout,
    ).to(device)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    if missing:
        raise RuntimeError(
            f"Missing checkpoint parameters: {missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint parameters: {unexpected}"
        )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"  Parameters: {parameter_count:,}"
    )

    print(
        "  ✓ Frozen state_dict loaded successfully"
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print()
    print("[4/6] Generating test predictions...")

    X_test_t = torch.from_numpy(
        X_test.astype(np.float32)
    )

    test_dataset = TensorDataset(
        X_test_t
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    prediction_start = time.time()

    test_pred_scaled = predict(
        model,
        test_loader,
        device,
    )

    prediction_seconds = (
        time.time() - prediction_start
    )

    expected_prediction_shape = (
        X_test.shape[0],
        TOUT,
        NUM_TARGETS,
    )

    if test_pred_scaled.shape != expected_prediction_shape:
        raise ValueError(
            f"Unexpected prediction shape: "
            f"{test_pred_scaled.shape}; "
            f"expected {expected_prediction_shape}"
        )

    if not np.all(
        np.isfinite(test_pred_scaled)
    ):
        raise ValueError(
            "Predictions contain non-finite values."
        )

    print(
        f"  Predictions: {test_pred_scaled.shape}"
    )

    print(
        f"  Prediction runtime: "
        f"{prediction_seconds:.2f} seconds"
    )

    # ========================================================
    # REAL UNITS
    # ========================================================

    print()
    print("[5/6] Converting to real units and calculating metrics...")

    Y_test_real = inverse_transform_targets(
        Y_test,
        target_scaler,
    )

    test_pred_real = inverse_transform_targets(
        test_pred_scaled,
        target_scaler,
    )

    test_metrics = calculate_metrics(
        Y_test_real,
        test_pred_real,
        Y_mask_test,
    )

    # ========================================================
    # SAVE ARTIFACTS
    # ========================================================

    print()
    print("[6/6] Saving final evaluation artifacts...")

    np.save(
        PREDICTIONS_PATH,
        test_pred_real,
    )

    np.save(
        TRUTH_PATH,
        Y_test_real,
    )

    np.save(
        MASK_PATH,
        Y_mask_test,
    )

    # --------------------------------------------------------
    # Standardized metrics format
    # --------------------------------------------------------

    standardized_metrics = {
        "model": "lstm",
        "model_type": "LSTMForecaster",
        "evaluation_type": "final_test_real_units",
        "overall": test_metrics["overall"],
        "per_target": test_metrics["per_target"],
        "horizon_wise": test_metrics["horizon_wise"],
        "metadata": {
            "direct_target_prediction": True,
            "residual_learning": False,
            "dataset": "official_phase7_test",
            "features": NUM_FEATURES,
            "input_hours": TIN,
            "forecast_hours": TOUT,
            "targets": TARGET_NAMES,
            "test_samples": int(X_test.shape[0]),
            "checkpoint": str(CHECKPOINT_PATH),
            "best_epoch": checkpoint.get(
                "epoch"
            ),
            "best_validation_mse": checkpoint.get(
                "val_loss"
            ),
            "prediction_runtime_seconds":
                prediction_seconds,
        },
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            standardized_metrics,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL RESULTS
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
    print("PER TARGET")

    for target_name in TARGET_NAMES:

        metrics = test_metrics[
            "per_target"
        ][target_name]

        print()
        print(target_name)

        print(
            f"  MAE : {metrics['MAE']:.4f}"
        )

        print(
            f"  RMSE: {metrics['RMSE']:.4f}"
        )

        print(
            f"  R²  : {metrics['R2']:.4f}"
        )

    print()
    print("HORIZON-WISE")

    for horizon, metrics in test_metrics[
        "horizon_wise"
    ].items():

        print(
            f"{horizon} | "
            f"MAE={metrics['MAE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"R²={metrics['R2']:.4f}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Frozen checkpoint:")
    print(
        f"  {CHECKPOINT_PATH}"
    )

    print()
    print("Saved:")
    print(
        f"  Predictions : {PREDICTIONS_PATH}"
    )

    print(
        f"  Truth       : {TRUTH_PATH}"
    )

    print(
        f"  Mask        : {MASK_PATH}"
    )

    print(
        f"  Metrics     : {METRICS_PATH}"
    )

    print()
    print(
        f"Evaluation runtime: "
        f"{time.time() - total_start:.2f} seconds"
    )

    print()
    print("RETRAINING: NO")
    print("MODEL: FROZEN")
    print("DATASET: OFFICIAL PHASE 7 TEST")
    print("EVALUATION: REAL-UNIT MASKED")
    print("RESEARCH CONTRACT: PASSED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()