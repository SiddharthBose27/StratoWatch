"""
StratoWatch — Final Single-Site XGBoost Evaluation

Evaluation-only script.

Research contract:
    Input:
        Official Phase 7 test split
        24 hours × 134 features
        flattened to 3216 features

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
    This script MUST NOT retrain the models.

    It loads the 12 frozen XGBoost artifacts:

        outputs/final_baselines/xgboost/models/
            xgboost_h1_O3.joblib
            xgboost_h1_NO2.joblib
            ...
            xgboost_h6_O3.joblib
            xgboost_h6_NO2.joblib
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np


# ============================================================
# SHARED DATA LOADER
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
    / "xgboost"
)

MODEL_DIR = OUTPUT_DIR / "models"

METRICS_PATH = OUTPUT_DIR / "test_metrics.json"

PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.npy"
TRUTH_PATH = OUTPUT_DIR / "test_truth.npy"
MASK_PATH = OUTPUT_DIR / "test_mask.npy"


# ============================================================
# FROZEN RESEARCH CONTRACT
# ============================================================

TIN = 24
TOUT = 6
NUM_FEATURES = 134
NUM_TARGETS = 2

FLATTENED_INPUT_SIZE = TIN * NUM_FEATURES
OUTPUT_SIZE = TOUT * NUM_TARGETS

TARGET_NAMES = [
    "O3",
    "NO2",
]


# ============================================================
# METRICS
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
                y_pred[valid] - y_true[valid]
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
                    y_pred[valid] - y_true[valid]
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
        (true_values - pred_values) ** 2
    )

    sst = np.sum(
        (true_values - mean_true) ** 2
    )

    if sst <= 0:
        return float("nan")

    return float(
        1.0 - (sse / sst)
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> Dict:
    """
    Standard API evaluation contract.

    Returns:
        overall
        per_target
        horizon_wise
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Truth/prediction shape mismatch: "
            f"{y_true.shape} vs {y_pred.shape}"
        )

    if y_true.shape != mask.shape:
        raise ValueError(
            f"Truth/mask shape mismatch: "
            f"{y_true.shape} vs {mask.shape}"
        )

    expected_shape = (
        y_true.shape[0],
        TOUT,
        NUM_TARGETS,
    )

    if y_true.shape != expected_shape:
        raise ValueError(
            f"Expected evaluation shape {expected_shape}, "
            f"got {y_true.shape}"
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = {
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

    per_target = {}

    for target_idx, target_name in enumerate(TARGET_NAMES):

        target_true = y_true[:, :, target_idx]
        target_pred = y_pred[:, :, target_idx]
        target_mask = mask[:, :, target_idx]

        per_target[target_name] = {
            "MAE": masked_mae(
                target_true,
                target_pred,
                target_mask,
            ),
            "RMSE": masked_rmse(
                target_true,
                target_pred,
                target_mask,
            ),
            "R2": masked_r2(
                target_true,
                target_pred,
                target_mask,
            ),
        }

    # --------------------------------------------------------
    # Horizon-wise
    # --------------------------------------------------------

    horizon_wise = {}

    for horizon_idx in range(TOUT):

        horizon_true = y_true[:, horizon_idx, :]
        horizon_pred = y_pred[:, horizon_idx, :]
        horizon_mask = mask[:, horizon_idx, :]

        horizon_wise[
            f"horizon_{horizon_idx + 1}"
        ] = {
            "MAE": masked_mae(
                horizon_true,
                horizon_pred,
                horizon_mask,
            ),
            "RMSE": masked_rmse(
                horizon_true,
                horizon_pred,
                horizon_mask,
            ),
            "R2": masked_r2(
                horizon_true,
                horizon_pred,
                horizon_mask,
            ),
        }

    return {
        "overall": overall,
        "per_target": per_target,
        "horizon_wise": horizon_wise,
    }


# ============================================================
# MODEL PATH HELPER
# ============================================================

def model_path(
    horizon: int,
    target_name: str,
) -> Path:
    return (
        MODEL_DIR
        / f"xgboost_h{horizon}_{target_name}.joblib"
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    start_time = time.time()

    print()
    print("=" * 70)
    print("STRATOWATCH — XGBOOST FINAL EVALUATION")
    print("=" * 70)

    # ========================================================
    # 1. LOAD OFFICIAL PHASE 7 TEST DATA
    # ========================================================

    print()
    print("[1/5] Loading official Phase 7 test data...")

    splits = load_splits()

    X_test = splits["X_test_flat"]
    Y_test_scaled = splits["Y_test"]
    Y_mask_test = splits["Y_mask_test"]
    target_scaler = splits["target_scaler"]

    print(f"  X_test : {X_test.shape}")
    print(f"  Y_test : {Y_test_scaled.shape}")
    print(f"  Mask   : {Y_mask_test.shape}")

    # --------------------------------------------------------
    # Contract checks
    # --------------------------------------------------------

    if X_test.ndim != 2:
        raise ValueError(
            f"Expected X_test to be 2D, got {X_test.shape}"
        )

    if X_test.shape[1] != FLATTENED_INPUT_SIZE:
        raise ValueError(
            f"Expected {FLATTENED_INPUT_SIZE} flattened input "
            f"features, got {X_test.shape[1]}"
        )

    if Y_test_scaled.shape != (
        X_test.shape[0],
        TOUT,
        NUM_TARGETS,
    ):
        raise ValueError(
            "Y_test shape does not match the frozen contract: "
            f"got {Y_test_scaled.shape}"
        )

    if Y_mask_test.shape != Y_test_scaled.shape:
        raise ValueError(
            f"Mask shape {Y_mask_test.shape} does not match "
            f"Y_test shape {Y_test_scaled.shape}"
        )

    if not np.all(np.isfinite(X_test)):
        raise ValueError(
            "X_test contains non-finite values."
        )

    if not np.all(np.isfinite(Y_test_scaled)):
        raise ValueError(
            "Y_test contains non-finite values."
        )

    # ========================================================
    # 2. LOAD ALL 12 FROZEN MODELS
    # ========================================================

    print()
    print("[2/5] Loading 12 frozen XGBoost models...")

    models = {}

    for horizon in range(1, TOUT + 1):

        for target_name in TARGET_NAMES:

            path = model_path(
                horizon,
                target_name,
            )

            if not path.exists():
                raise FileNotFoundError(
                    "Frozen XGBoost model not found:\n"
                    f"{path}"
                )

            model = joblib.load(path)

            # ------------------------------------------------
            # Validate input contract
            # ------------------------------------------------

            if not hasattr(
                model,
                "n_features_in_",
            ):
                raise ValueError(
                    f"Model does not expose n_features_in_: "
                    f"{path}"
                )

            if (
                model.n_features_in_
                != FLATTENED_INPUT_SIZE
            ):
                raise ValueError(
                    f"Input contract mismatch for {path}: "
                    f"expected {FLATTENED_INPUT_SIZE}, "
                    f"got {model.n_features_in_}"
                )

            models[
                (horizon, target_name)
            ] = model

            print(
                f"  Loaded H+{horizon} {target_name}: "
                f"{path.name}"
            )

    if len(models) != OUTPUT_SIZE:
        raise ValueError(
            f"Expected {OUTPUT_SIZE} XGBoost models, "
            f"loaded {len(models)}"
        )

    # ========================================================
    # 3. GENERATE FROZEN-MODEL PREDICTIONS
    # ========================================================

    print()
    print("[3/5] Generating test predictions...")

    prediction_start = time.time()

    predictions_scaled = np.zeros(
        (
            X_test.shape[0],
            TOUT,
            NUM_TARGETS,
        ),
        dtype=np.float64,
    )

    for horizon in range(1, TOUT + 1):

        for target_idx, target_name in enumerate(
            TARGET_NAMES
        ):

            model = models[
                (horizon, target_name)
            ]

            output = model.predict(
                X_test
            )

            output = np.asarray(
                output,
                dtype=np.float64,
            )

            if output.ndim != 1:
                output = output.reshape(-1)

            if output.shape[0] != X_test.shape[0]:
                raise ValueError(
                    f"Prediction length mismatch for "
                    f"H+{horizon} {target_name}: "
                    f"expected {X_test.shape[0]}, "
                    f"got {output.shape[0]}"
                )

            if not np.all(np.isfinite(output)):
                raise ValueError(
                    f"Non-finite predictions from "
                    f"H+{horizon} {target_name}"
                )

            predictions_scaled[
                :,
                horizon - 1,
                target_idx,
            ] = output

    prediction_seconds = (
        time.time() - prediction_start
    )

    print(
        f"  Predictions: "
        f"{predictions_scaled.shape}"
    )

    print(
        f"  Prediction runtime: "
        f"{prediction_seconds:.2f} seconds"
    )

    # ========================================================
    # 4. CONVERT TO REAL UNITS + EVALUATE
    # ========================================================

    print()
    print("[4/5] Converting to real units and calculating metrics...")

    y_true_real = inverse_transform_targets(
        Y_test_scaled,
        target_scaler,
    )

    y_pred_real = inverse_transform_targets(
        predictions_scaled,
        target_scaler,
    )

    if not np.all(np.isfinite(y_true_real)):
        raise ValueError(
            "Inverse-transformed truth contains "
            "non-finite values."
        )

    if not np.all(np.isfinite(y_pred_real)):
        raise ValueError(
            "Inverse-transformed predictions contain "
            "non-finite values."
        )

    metrics = calculate_metrics(
        y_true_real,
        y_pred_real,
        Y_mask_test,
    )

    # ========================================================
    # 5. SAVE FINAL EVALUATION ARTIFACTS
    # ========================================================

    print()
    print("[5/5] Saving final evaluation artifacts...")

    np.save(
        PREDICTIONS_PATH,
        y_pred_real,
    )

    np.save(
        TRUTH_PATH,
        y_true_real,
    )

    np.save(
        MASK_PATH,
        Y_mask_test,
    )

    evaluation_seconds = (
        time.time() - start_time
    )

    metrics_output = {
        "model": "xgboost",
        "evaluation_type": "final_test_real_units",

        "overall": metrics["overall"],

        "per_target": metrics["per_target"],

        "horizon_wise": metrics["horizon_wise"],

        "metadata": {
            "model_type": "XGBRegressor",
            "number_of_regressors": OUTPUT_SIZE,
            "target_mode": "direct_target_prediction",
            "residual_learning": False,
            "dataset": "official_phase7_test",
            "input_window_hours": TIN,
            "forecast_horizon_hours": TOUT,
            "feature_count": NUM_FEATURES,
            "flattened_input_features": FLATTENED_INPUT_SIZE,
            "target_count": NUM_TARGETS,
            "targets": TARGET_NAMES,
            "test_samples": int(X_test.shape[0]),
            "model_directory": str(MODEL_DIR),
            "prediction_seconds": prediction_seconds,
            "evaluation_seconds": evaluation_seconds,
        },
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_output,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("XGBOOST — FINAL TEST RESULTS")
    print("=" * 70)

    print()
    print("OVERALL")

    print(
        f"MAE : "
        f"{metrics['overall']['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{metrics['overall']['RMSE']:.4f}"
    )

    print(
        f"R²  : "
        f"{metrics['overall']['R2']:.4f}"
    )

    print()
    print("PER TARGET")

    for target_name in TARGET_NAMES:

        target_metrics = metrics[
            "per_target"
        ][target_name]

        print()
        print(target_name)

        print(
            f"  MAE : "
            f"{target_metrics['MAE']:.4f}"
        )

        print(
            f"  RMSE: "
            f"{target_metrics['RMSE']:.4f}"
        )

        print(
            f"  R²  : "
            f"{target_metrics['R2']:.4f}"
        )

    print()
    print("HORIZON-WISE")

    for horizon, horizon_metrics in (
        metrics["horizon_wise"].items()
    ):

        print(
            f"{horizon} | "
            f"MAE={horizon_metrics['MAE']:.4f} | "
            f"RMSE={horizon_metrics['RMSE']:.4f} | "
            f"R²={horizon_metrics['R2']:.4f}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Frozen models:")
    print(f"  {MODEL_DIR}")

    print()
    print("Saved:")
    print(f"  Predictions : {PREDICTIONS_PATH}")
    print(f"  Truth       : {TRUTH_PATH}")
    print(f"  Mask        : {MASK_PATH}")
    print(f"  Metrics     : {METRICS_PATH}")

    print()
    print(
        f"Evaluation runtime: "
        f"{evaluation_seconds:.2f} seconds"
    )

    print()
    print("REGRESSORS LOADED: 12")
    print("RETRAINING: NO")
    print("MODEL: FROZEN")
    print("DATASET: OFFICIAL PHASE 7 TEST")
    print("EVALUATION: REAL-UNIT MASKED")
    print("RESEARCH CONTRACT: PASSED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()