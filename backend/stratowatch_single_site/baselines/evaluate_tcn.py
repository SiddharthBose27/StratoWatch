from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .common_data import (
    load_splits,
    inverse_transform_targets,
)


# ============================================================================
# STRATOWATCH — TCN FINAL EVALUATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
SINGLE_SITE_DIR = BASE_DIR.parent

OUTPUT_DIR = (
    SINGLE_SITE_DIR
    / "outputs"
    / "final_baselines"
    / "tcn"
)

CHECKPOINT_PATH = OUTPUT_DIR / "tcn_best.pt"

PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.npy"
TRUTH_PATH = OUTPUT_DIR / "test_truth.npy"
MASK_PATH = OUTPUT_DIR / "test_mask.npy"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"


# ============================================================================
# FROZEN RESEARCH CONTRACT
# ============================================================================

INPUT_SIZE = 134
TIN = 24
TOUT = 6
OUTPUT_SIZE = 2

CHANNELS = [64, 64, 64]
KERNEL_SIZE = 3
DROPOUT = 0.20

TARGET_NAMES = ["O3", "NO2"]


# ============================================================================
# TCN BUILDING BLOCK
# ============================================================================

class Chomp1d(nn.Module):

    def __init__(self, chomp_size: int):
        super().__init__()

        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp1 = Chomp1d(padding)

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp2 = Chomp1d(padding)

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        if n_inputs != n_outputs:

            self.downsample = nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size=1,
            )

        else:

            self.downsample = None

        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:

            residual = self.downsample(x)

        else:

            residual = x

        return self.relu(out + residual)


# ============================================================================
# TCN FORECASTER
# ============================================================================

class TCNForecaster(nn.Module):

    def __init__(
        self,
        input_size: int,
        channels: list[int],
        kernel_size: int,
        output_size: int,
        forecast_horizon: int,
        dropout: float,
    ):
        super().__init__()

        layers = []

        for i, out_channels in enumerate(channels):

            in_channels = (
                input_size
                if i == 0
                else channels[i - 1]
            )

            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # IMPORTANT:
        # The training checkpoint uses the attribute name "tcn".
        #
        # Therefore the state_dict contains:
        #
        #   tcn.0.conv1.weight
        #   tcn.0.conv2.weight
        #   tcn.0.downsample.weight
        #
        # and NOT:
        #
        #   network.0...
        #
        self.tcn = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Linear(
                channels[-1],
                64,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                forecast_horizon * output_size,
            ),
        )

        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Input:
        #   (B, T, F)
        #
        # Conv1d:
        #   (B, F, T)

        x = x.transpose(1, 2)

        for block in self.tcn:
            x = block(x)

        # Latest timestep representation
        x = x[:, :, -1]

        x = self.head(x)

        x = x.view(
            x.shape[0],
            self.forecast_horizon,
            self.output_size,
        )

        return x


# ============================================================================
# CHECKPOINT VALIDATION
# ============================================================================

def validate_checkpoint_config(
    checkpoint: Dict,
) -> None:

    if "model_config" not in checkpoint:

        raise RuntimeError(
            "TCN checkpoint does not contain "
            "'model_config'."
        )

    config = checkpoint["model_config"]

    expected = {
        "input_size": INPUT_SIZE,
        "channels": CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "output_size": OUTPUT_SIZE,
        "forecast_horizon": TOUT,
        "dropout": DROPOUT,
    }

    print("\nCheckpoint model configuration:")

    for key, expected_value in expected.items():

        if key not in config:

            raise RuntimeError(
                f"Checkpoint missing "
                f"model_config['{key}']."
            )

        actual_value = config[key]

        print(
            f"  {key}: "
            f"checkpoint={actual_value} "
            f"expected={expected_value}"
        )

        if actual_value != expected_value:

            raise RuntimeError(
                "\nTCN MODEL CONTRACT MISMATCH\n"
                f"Parameter: {key}\n"
                f"Checkpoint: {actual_value}\n"
                f"Expected: {expected_value}"
            )

    contract = checkpoint.get(
        "research_contract"
    )

    if contract is not None:

        print("\nCheckpoint research contract:")

        expected_contract = {
            "features": INPUT_SIZE,
            "tin": TIN,
            "tout": TOUT,
            "targets": TARGET_NAMES,
        }

        for key, expected_value in (
            expected_contract.items()
        ):

            if key not in contract:

                raise RuntimeError(
                    f"Checkpoint research_contract "
                    f"missing '{key}'."
                )

            actual_value = contract[key]

            print(
                f"  {key}: "
                f"checkpoint={actual_value} "
                f"expected={expected_value}"
            )

            if actual_value != expected_value:

                raise RuntimeError(
                    "\nTCN RESEARCH CONTRACT MISMATCH\n"
                    f"Parameter: {key}\n"
                    f"Checkpoint: {actual_value}\n"
                    f"Expected: {expected_value}"
                )


# ============================================================================
# METRICS
# ============================================================================

def safe_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if len(y_true) == 0:
        return float("nan")

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot <= 1e-12:
        return float("nan")

    return float(
        1.0 - ss_res / ss_tot
    )


def calculate_metrics(
    predictions: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> Tuple[Dict, Dict, Dict]:

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    mask = np.asarray(mask).astype(bool)

    if predictions.shape != truth.shape:

        raise ValueError(
            "Prediction/truth shape mismatch: "
            f"{predictions.shape} vs "
            f"{truth.shape}"
        )

    if truth.shape != mask.shape:

        raise ValueError(
            "Truth/mask shape mismatch: "
            f"{truth.shape} vs "
            f"{mask.shape}"
        )

    # ------------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------------

    valid = mask

    y_true = truth[valid]
    y_pred = predictions[valid]

    if len(y_true) == 0:

        raise RuntimeError(
            "No valid target values found."
        )

    error = y_pred - y_true

    overall = {
        "mae": float(
            np.mean(np.abs(error))
        ),
        "rmse": float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),
        "r2": float(
            safe_r2(
                y_true,
                y_pred,
            )
        ),
    }

    # ------------------------------------------------------------------------
    # Per target
    # ------------------------------------------------------------------------

    per_target = {}

    for idx, target_name in enumerate(
        TARGET_NAMES
    ):

        target_mask = mask[:, :, idx]

        target_true = truth[:, :, idx][
            target_mask
        ]

        target_pred = predictions[:, :, idx][
            target_mask
        ]

        if len(target_true) == 0:

            per_target[target_name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
            }

            continue

        target_error = (
            target_pred - target_true
        )

        per_target[target_name] = {
            "mae": float(
                np.mean(
                    np.abs(target_error)
                )
            ),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        target_error ** 2
                    )
                )
            ),
            "r2": float(
                safe_r2(
                    target_true,
                    target_pred,
                )
            ),
        }

    # ------------------------------------------------------------------------
    # Horizon-wise
    # ------------------------------------------------------------------------

    horizon_wise = {}

    for horizon_idx in range(TOUT):

        horizon_name = (
            f"H+{horizon_idx + 1}"
        )

        horizon_mask = mask[
            :,
            horizon_idx,
            :,
        ]

        horizon_true = truth[
            :,
            horizon_idx,
            :,
        ][horizon_mask]

        horizon_pred = predictions[
            :,
            horizon_idx,
            :,
        ][horizon_mask]

        if len(horizon_true) == 0:

            horizon_wise[horizon_name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
            }

            continue

        horizon_error = (
            horizon_pred - horizon_true
        )

        horizon_wise[horizon_name] = {
            "mae": float(
                np.mean(
                    np.abs(horizon_error)
                )
            ),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        horizon_error ** 2
                    )
                )
            ),
            "r2": float(
                safe_r2(
                    horizon_true,
                    horizon_pred,
                )
            ),
        }

    return (
        overall,
        per_target,
        horizon_wise,
    )


# ============================================================================
# JSON CLEANER
# ============================================================================

def clean_for_json(obj):

    if isinstance(obj, dict):

        return {
            str(k): clean_for_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            clean_for_json(v)
            for v in obj
        ]

    if isinstance(obj, float):

        if np.isnan(obj) or np.isinf(obj):
            return None

        return obj

    if isinstance(obj, np.floating):

        value = float(obj)

        if np.isnan(value) or np.isinf(value):
            return None

        return value

    if isinstance(obj, np.integer):

        return int(obj)

    return obj


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("STRATOWATCH — TCN FINAL EVALUATION")
    print("=" * 70)

    print("\nEVALUATION ONLY")
    print("RETRAINING: NO")
    print("MODEL: FROZEN")
    print("DATASET: OFFICIAL PHASE 7 TEST")
    print("EVALUATION: REAL-UNIT MASKED")

    device = torch.device("cpu")

    print(f"\nDevice: {device}")

    # ========================================================================
    # 1. CHECKPOINT
    # ========================================================================

    print(
        "\n[1/6] Loading frozen TCN checkpoint..."
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Frozen TCN checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Unexpected checkpoint format."
        )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    validate_checkpoint_config(
        checkpoint
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    # ========================================================================
    # 2. DATA
    # ========================================================================

    print(
        "\n[2/6] Loading official Phase 7 test data..."
    )

    splits = load_splits()

    X_test = splits["X_test"]
    Y_test = splits["Y_test"]
    Y_mask_test = splits["Y_mask_test"]

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"Y_test:  {Y_test.shape}"
    )

    print(
        f"Mask:    {Y_mask_test.shape}"
    )

    if X_test.shape[1:] != (
        TIN,
        INPUT_SIZE,
    ):

        raise RuntimeError(
            "\nTCN INPUT CONTRACT FAILED\n"
            f"Expected: (N, {TIN}, {INPUT_SIZE})\n"
            f"Actual:   {X_test.shape}"
        )

    if Y_test.shape[1:] != (
        TOUT,
        OUTPUT_SIZE,
    ):

        raise RuntimeError(
            "\nTCN TARGET CONTRACT FAILED\n"
            f"Expected: (N, {TOUT}, {OUTPUT_SIZE})\n"
            f"Actual:   {Y_test.shape}"
        )

    if Y_mask_test.shape != Y_test.shape:

        raise RuntimeError(
            "\nTCN MASK CONTRACT FAILED\n"
            f"Y_test: {Y_test.shape}\n"
            f"Mask:   {Y_mask_test.shape}"
        )

    if not np.isfinite(X_test).all():

        raise RuntimeError(
            "X_test contains NaN or infinite values."
        )

    if not np.isfinite(Y_test).all():

        raise RuntimeError(
            "Y_test contains NaN or infinite values."
        )

    if not np.all(
        np.isin(
            np.unique(Y_mask_test),
            [0, 1],
        )
    ):

        raise RuntimeError(
            "Target mask must contain only 0/1."
        )

    # ========================================================================
    # 3. MODEL
    # ========================================================================

    print(
        "\n[3/6] Reconstructing exact TCN architecture..."
    )

    model = TCNForecaster(
        input_size=INPUT_SIZE,
        channels=CHANNELS,
        kernel_size=KERNEL_SIZE,
        output_size=OUTPUT_SIZE,
        forecast_horizon=TOUT,
        dropout=DROPOUT,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nArchitecture:")
    print(
        f"  Input size:       {INPUT_SIZE}"
    )
    print(
        f"  History length:   {TIN}"
    )
    print(
        f"  Channels:         {CHANNELS}"
    )
    print(
        f"  Kernel size:      {KERNEL_SIZE}"
    )
    print(
        f"  Dilations:        {[1, 2, 4]}"
    )
    print(
        f"  Dropout:          {DROPOUT}"
    )
    print(
        f"  Forecast horizon: {TOUT}"
    )
    print(
        f"  Output size:      {OUTPUT_SIZE}"
    )
    print(
        f"  Parameters:       {parameter_count:,}"
    )

    # ========================================================================
    # 4. LOAD STATE DICT
    # ========================================================================

    print(
        "\n[4/6] Loading frozen state_dict..."
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    # First perform an explicit key check.
    model_keys = set(
        model.state_dict().keys()
    )

    checkpoint_keys = set(
        state_dict.keys()
    )

    missing_keys = sorted(
        model_keys - checkpoint_keys
    )

    unexpected_keys = sorted(
        checkpoint_keys - model_keys
    )

    if missing_keys or unexpected_keys:

        print(
            "\nState-dict mismatch detected:"
        )

        if missing_keys:

            print("Missing keys:")

            for key in missing_keys:
                print(f"  - {key}")

        if unexpected_keys:

            print("Unexpected keys:")

            for key in unexpected_keys:
                print(f"  - {key}")

        raise RuntimeError(
            "\nTCN state_dict does not exactly "
            "match the reconstructed architecture."
        )

    # Exact load.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "State dict loaded successfully."
    )

    print(
        "Strict architecture validation: PASSED"
    )

    model.eval()

    # ========================================================================
    # 5. PREDICTIONS
    # ========================================================================

    print(
        "\n[5/6] Generating predictions..."
    )

    X_test_tensor = torch.as_tensor(
        X_test,
        dtype=torch.float32,
    )

    batch_size = 256

    predictions_scaled = []

    start_time = time.time()

    with torch.no_grad():

        for start in range(
            0,
            len(X_test_tensor),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(X_test_tensor),
            )

            batch = X_test_tensor[
                start:end
            ].to(device)

            output = model(batch)

            predictions_scaled.append(
                output.cpu().numpy()
            )

    prediction_runtime = (
        time.time() - start_time
    )

    predictions_scaled = np.concatenate(
        predictions_scaled,
        axis=0,
    )

    print(
        f"Predictions shape: "
        f"{predictions_scaled.shape}"
    )

    print(
        f"Prediction runtime: "
        f"{prediction_runtime:.2f} sec"
    )

    if predictions_scaled.shape != Y_test.shape:

        raise RuntimeError(
            "\nTCN OUTPUT CONTRACT FAILED\n"
            f"Expected: {Y_test.shape}\n"
            f"Actual:   {predictions_scaled.shape}"
        )

    if not np.isfinite(
        predictions_scaled
    ).all():

        raise RuntimeError(
            "Predictions contain NaN "
            "or infinite values."
        )

    # ------------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------------

    print(
    "\nInverse-transforming predictions and truth..."
    )

    target_scaler = splits["target_scaler"]

    predictions_real = (
        inverse_transform_targets(
            predictions_scaled,
            target_scaler,
        )
    )

    truth_real = (
        inverse_transform_targets(
            Y_test,
            target_scaler,
        )
    )

    predictions_real = np.asarray(
        predictions_real,
        dtype=np.float64,
    )

    truth_real = np.asarray(
        truth_real,
        dtype=np.float64,
    )

    print(
        f"Real predictions shape: "
        f"{predictions_real.shape}"
    )

    print(
        f"Real truth shape: "
        f"{truth_real.shape}"
    )

    # ========================================================================
    # 6. METRICS
    # ========================================================================

    print(
        "\n[6/6] Computing final "
        "real-unit masked metrics..."
    )

    (
        overall,
        per_target,
        horizon_wise,
    ) = calculate_metrics(
        predictions=predictions_real,
        truth=truth_real,
        mask=Y_mask_test,
    )

    # ------------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TCN TEST RESULTS")
    print("=" * 70)

    print("\nOverall:")

    print(
        f"  MAE : {overall['mae']:.4f}"
    )

    print(
        f"  RMSE: {overall['rmse']:.4f}"
    )

    print(
        f"  R²  : {overall['r2']:.4f}"
    )

    print("\nPer target:")

    for target_name in TARGET_NAMES:

        metrics = per_target[
            target_name
        ]

        print(
            f"\n  {target_name}:"
        )

        print(
            f"    MAE : {metrics['mae']:.4f}"
        )

        print(
            f"    RMSE: {metrics['rmse']:.4f}"
        )

        print(
            f"    R²  : {metrics['r2']:.4f}"
        )

    print("\nHorizon-wise:")

    for horizon_name, metrics in (
        horizon_wise.items()
    ):

        print(
            f"  {horizon_name}: "
            f"MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} "
            f"R²={metrics['r2']:.4f}"
        )

    # ========================================================================
    # SAVE
    # ========================================================================

    print(
        "\nSaving evaluation artifacts..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        PREDICTIONS_PATH,
        predictions_real,
    )

    np.save(
        TRUTH_PATH,
        truth_real,
    )

    np.save(
        MASK_PATH,
        Y_mask_test.astype(np.uint8),
    )

    metrics_payload = {

        "model": "tcn",

        "evaluation_type":
            "final_test_real_units",

        "overall": overall,

        "per_target": per_target,

        "horizon_wise": horizon_wise,

        "metadata": {

            "dataset":
                "official_phase_7_test",

            "evaluation":
                "real_unit_masked",

            "retraining": False,

            "model_frozen": True,

            "input_features":
                INPUT_SIZE,

            "history_hours":
                TIN,

            "forecast_horizon_hours":
                TOUT,

            "targets":
                TARGET_NAMES,

            "channels":
                CHANNELS,

            "kernel_size":
                KERNEL_SIZE,

            "dilations":
                [1, 2, 4],

            "dropout":
                DROPOUT,

            "test_samples":
                int(X_test.shape[0]),

            "x_test_shape":
                list(X_test.shape),

            "y_test_shape":
                list(Y_test.shape),

            "mask_shape":
                list(Y_mask_test.shape),

            "parameter_count":
                int(parameter_count),

            "checkpoint":
                str(CHECKPOINT_PATH),

            "prediction_runtime_seconds":
                float(prediction_runtime),

            "research_contract": {

                "features":
                    INPUT_SIZE,

                "tin":
                    TIN,

                "tout":
                    TOUT,

                "targets":
                    TARGET_NAMES,
            },
        },
    }

    metrics_payload = clean_for_json(
        metrics_payload
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_payload,
            f,
            indent=2,
        )

    # ========================================================================
    # COMPLETE
    # ========================================================================

    print("\nSaved:")

    print(
        f"  Predictions: "
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"  Truth:       "
        f"{TRUTH_PATH}"
    )

    print(
        f"  Mask:        "
        f"{MASK_PATH}"
    )

    print(
        f"  Metrics:     "
        f"{METRICS_PATH}"
    )

    print("\n" + "=" * 70)
    print("TCN FINAL EVALUATION COMPLETE")
    print("=" * 70)

    print("\nRETRAINING: NO")
    print("MODEL: FROZEN")
    print("DATASET: OFFICIAL PHASE 7 TEST")
    print("EVALUATION: REAL-UNIT MASKED")
    print("RESEARCH CONTRACT: PASSED")


if __name__ == "__main__":
    main()