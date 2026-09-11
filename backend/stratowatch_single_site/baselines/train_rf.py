"""
StratoWatch — Final Single-Site Random Forest Baseline

Research contract:
    Input:
        24 hours × 134 features
        flattened to 3216 features

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

The Random Forest is trained directly on the target values.
It is NOT a residual-learning model.

This script is intended for the final research comparison against:
    - Forecast-only baseline
    - XGBoost
    - LSTM
    - TCN
    - Transformer
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
    / "random_forest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RANDOM FOREST CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = "sqrt"

N_JOBS = -1


# ============================================================
# DATA CONTRACT
# ============================================================

TIN = 24
TOUT = 6
NUM_FEATURES = 134
NUM_TARGETS = 2

OUTPUT_SIZE = TOUT * NUM_TARGETS

TARGET_NAMES = [
    "O3",
    "NO2",
]


# ============================================================
# METRIC HELPERS
# ============================================================

def masked_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute globally masked MAE."""

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
    """Compute globally masked RMSE."""

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
    """
    Compute globally masked R².

    Formula:
        R² = 1 - SSE / SST
    """

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
    Calculate:
        - overall metrics
        - O3 metrics
        - NO2 metrics
        - horizon-wise metrics
    """

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

    results["targets"] = {}

    for target_idx, target_name in enumerate(TARGET_NAMES):

        target_true = y_true[:, :, target_idx]
        target_pred = y_pred[:, :, target_idx]
        target_mask = mask[:, :, target_idx]

        results["targets"][target_name] = {
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

    results["horizons"] = {}

    for horizon_idx in range(y_true.shape[1]):

        h_true = y_true[:, horizon_idx, :]
        h_pred = y_pred[:, horizon_idx, :]
        h_mask = mask[:, horizon_idx, :]

        results["horizons"][
            f"H+{horizon_idx + 1}"
        ] = {
            "MAE": masked_mae(
                h_true,
                h_pred,
                h_mask,
            ),
            "RMSE": masked_rmse(
                h_true,
                h_pred,
                h_mask,
            ),
            "R2": masked_r2(
                h_true,
                h_pred,
                h_mask,
            ),
        }

    return results


# ============================================================
# TARGET RESHAPING
# ============================================================

def flatten_targets(
    Y: np.ndarray,
) -> np.ndarray:
    """
    Convert:

        (N, 6, 2)

    into:

        (N, 12)

    Output ordering:

        H+1 O3
        H+1 NO2
        H+2 O3
        H+2 NO2
        ...
        H+6 O3
        H+6 NO2
    """

    if Y.ndim != 3:
        raise ValueError(
            f"Expected Y to have 3 dimensions, got {Y.shape}"
        )

    if Y.shape[1] != TOUT:
        raise ValueError(
            f"Expected Tout={TOUT}, got {Y.shape[1]}"
        )

    if Y.shape[2] != NUM_TARGETS:
        raise ValueError(
            f"Expected {NUM_TARGETS} targets, got {Y.shape[2]}"
        )

    return Y.reshape(
        Y.shape[0],
        OUTPUT_SIZE,
    )


def unflatten_predictions(
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Convert:

        (N, 12)

    back into:

        (N, 6, 2)
    """

    if predictions.ndim != 2:
        raise ValueError(
            f"Expected predictions to have 2 dimensions, "
            f"got {predictions.shape}"
        )

    if predictions.shape[1] != OUTPUT_SIZE:
        raise ValueError(
            f"Expected {OUTPUT_SIZE} outputs, "
            f"got {predictions.shape[1]}"
        )

    return predictions.reshape(
        predictions.shape[0],
        TOUT,
        NUM_TARGETS,
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def build_model() -> RandomForestRegressor:
    """
    Build the final Random Forest.

    The model predicts all 12 horizon-target outputs jointly.
    """

    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("STRATOWATCH — FINAL SINGLE-SITE RANDOM FOREST")
    print("=" * 70)

    start_time = time.time()

    # ========================================================
    # LOAD OFFICIAL PHASE 7 DATA
    # ========================================================

    print()
    print("[1/6] Loading official Phase 7 dataset...")

    splits = load_splits()

    X_train = splits["X_train_flat"]
    X_val = splits["X_val_flat"]
    X_test = splits["X_test_flat"]

    Y_train_scaled = splits["Y_train"]
    Y_val_scaled = splits["Y_val"]
    Y_test_scaled = splits["Y_test"]

    Y_mask_train = splits["Y_mask_train"]
    Y_mask_val = splits["Y_mask_val"]
    Y_mask_test = splits["Y_mask_test"]

    target_scaler = splits["target_scaler"]

    # --------------------------------------------------------
    # Contract checks
    # --------------------------------------------------------

    if X_train.shape[1] != TIN * NUM_FEATURES:
        raise ValueError(
            "Training feature count does not match "
            f"24 × 134 = {TIN * NUM_FEATURES}."
        )

    if X_val.shape[1] != TIN * NUM_FEATURES:
        raise ValueError(
            "Validation feature count does not match "
            f"24 × 134 = {TIN * NUM_FEATURES}."
        )

    if X_test.shape[1] != TIN * NUM_FEATURES:
        raise ValueError(
            "Test feature count does not match "
            f"24 × 134 = {TIN * NUM_FEATURES}."
        )

    print()
    print("Dataset:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  X_test : {X_test.shape}")

    print()
    print("Targets:")
    print(f"  Y_train: {Y_train_scaled.shape}")
    print(f"  Y_val  : {Y_val_scaled.shape}")
    print(f"  Y_test : {Y_test_scaled.shape}")

    # ========================================================
    # PREPARE TARGETS
    # ========================================================

    print()
    print("[2/6] Preparing 6-hour × 2-target outputs...")

    Y_train_flat = flatten_targets(
        Y_train_scaled
    )

    Y_val_flat = flatten_targets(
        Y_val_scaled
    )

    Y_test_flat = flatten_targets(
        Y_test_scaled
    )

    print(
        f"  Y_train_flat: {Y_train_flat.shape}"
    )

    print(
        f"  Y_val_flat  : {Y_val_flat.shape}"
    )

    print(
        f"  Y_test_flat : {Y_test_flat.shape}"
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print()
    print("[3/6] Building Random Forest...")

    model = build_model()

    print()
    print("Random Forest configuration:")
    print(f"  n_estimators       : {N_ESTIMATORS}")
    print(f"  max_depth          : {MAX_DEPTH}")
    print(f"  min_samples_split  : {MIN_SAMPLES_SPLIT}")
    print(f"  min_samples_leaf   : {MIN_SAMPLES_LEAF}")
    print(f"  max_features       : {MAX_FEATURES}")
    print(f"  random_state       : {RANDOM_STATE}")
    print(f"  n_jobs             : {N_JOBS}")

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("[4/6] Training Random Forest...")
    print()
    print(
        "This may take several minutes because "
        "the model uses 30,438 samples × 3,216 features."
    )

    train_start = time.time()

    model.fit(
        X_train,
        Y_train_flat,
    )

    train_seconds = time.time() - train_start

    print()
    print(
        f"Training complete in "
        f"{train_seconds:.2f} seconds."
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    print()
    print("[5/6] Generating validation and test predictions...")

    val_pred_scaled_flat = model.predict(
        X_val
    )

    test_pred_scaled_flat = model.predict(
        X_test
    )

    val_pred_scaled = unflatten_predictions(
        val_pred_scaled_flat
    )

    test_pred_scaled = unflatten_predictions(
        test_pred_scaled_flat
    )

    print(
        f"  Validation predictions: "
        f"{val_pred_scaled.shape}"
    )

    print(
        f"  Test predictions      : "
        f"{test_pred_scaled.shape}"
    )

    # ========================================================
    # INVERSE TRANSFORM TO REAL UNITS
    # ========================================================

    print()
    print("Converting predictions to real units...")

    Y_val_real = inverse_transform_targets(
        Y_val_scaled,
        target_scaler,
    )

    Y_test_real = inverse_transform_targets(
        Y_test_scaled,
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
    print("[6/6] Calculating masked real-unit metrics...")

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
    # PRINT FINAL TEST RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("RANDOM FOREST — FINAL TEST RESULTS")
    print("=" * 70)

    print()
    print("OVERALL")
    print(
        f"MAE : {test_metrics['overall']['MAE']:.4f}"
    )
    print(
        f"RMSE: {test_metrics['overall']['RMSE']:.4f}"
    )
    print(
        f"R²  : {test_metrics['overall']['R2']:.4f}"
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
    # SAVE MODEL
    # ========================================================

    model_path = (
        OUTPUT_DIR
        / "random_forest_final.joblib"
    )

    joblib.dump(
        model,
        model_path,
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
    # SAVE METRICS
    # ========================================================

    metrics_output = {
        "model": "random_forest",
        "model_type": "RandomForestRegressor",
        "target_mode": "direct_target_prediction",
        "dataset": {
            "artifact": splits["data_path"],
            "features": NUM_FEATURES,
            "tin": TIN,
            "tout": TOUT,
            "targets": TARGET_NAMES,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
        },
        "configuration": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "training_seconds": train_seconds,
        "total_runtime_seconds": time.time() - start_time,
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
    # SAVE CONFIG / METADATA
    # ========================================================

    metadata = {
        "model": "random_forest",
        "research_contract": {
            "feature_count": NUM_FEATURES,
            "input_hours": TIN,
            "forecast_hours": TOUT,
            "target_count": NUM_TARGETS,
            "targets": TARGET_NAMES,
            "flattened_input_features": TIN * NUM_FEATURES,
            "flattened_outputs": TOUT * NUM_TARGETS,
            "direct_target_prediction": True,
            "residual_learning": False,
        },
        "dataset_artifact": splits["data_path"],
        "feature_schema": splits["feature_list_path"],
        "target_scaler": splits["target_scaler_path"],
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
    print("RANDOM FOREST COMPLETE")
    print("=" * 70)

    print()
    print("Saved:")

    print(
        f"  Model       : {model_path}"
    )

    print(
        f"  Metrics     : {metrics_path}"
    )

    print(
        f"  Predictions : "
        f"{OUTPUT_DIR / 'test_predictions.npy'}"
    )

    print(
        f"  Truth       : "
        f"{OUTPUT_DIR / 'test_truth.npy'}"
    )

    print(
        f"  Mask        : "
        f"{OUTPUT_DIR / 'test_mask.npy'}"
    )

    print(
        f"  Metadata    : "
        f"{OUTPUT_DIR / 'metadata.json'}"
    )

    print()
    print(
        f"Total runtime: "
        f"{time.time() - start_time:.2f} seconds"
    )

    print()
    print("RESEARCH CONTRACT: PASSED")
    print("RANDOM FOREST: COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()