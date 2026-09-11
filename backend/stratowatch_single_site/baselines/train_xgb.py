"""
StratoWatch — Final Single-Site XGBoost Baseline

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

The XGBoost model predicts the targets directly.
It does NOT use the old residual-learning pipeline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from xgboost import XGBRegressor


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
    / "xgboost"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 200
LEARNING_RATE = 0.05
MAX_DEPTH = 5

SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.2

N_JOBS = -1

TREE_METHOD = "hist"


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
    """Compute globally masked R²."""

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
    """Calculate overall, target-wise and horizon-wise metrics."""

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
# TARGET HELPERS
# ============================================================

def flatten_targets(
    Y: np.ndarray,
) -> np.ndarray:
    """
    Convert:

        (N, 6, 2)

    into:

        (N, 12)
    """

    if Y.ndim != 3:
        raise ValueError(
            f"Expected Y shape (N, 6, 2), got {Y.shape}"
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

    into:

        (N, 6, 2)
    """

    if predictions.ndim != 2:
        raise ValueError(
            f"Expected prediction matrix, got {predictions.shape}"
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
# MODEL
# ============================================================

def build_model() -> XGBRegressor:
    """Build one XGBoost regressor."""

    return XGBRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method=TREE_METHOD,
        objective="reg:squarederror",
        eval_metric="rmse",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("STRATOWATCH — FINAL SINGLE-SITE XGBOOST")
    print("=" * 70)

    total_start = time.time()

    # ========================================================
    # LOAD DATA
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
    # Contract validation
    # --------------------------------------------------------

    expected_input_features = TIN * NUM_FEATURES

    if X_train.shape[1] != expected_input_features:
        raise ValueError(
            f"Expected {expected_input_features} flattened features, "
            f"got {X_train.shape[1]}"
        )

    if X_val.shape[1] != expected_input_features:
        raise ValueError(
            f"Expected {expected_input_features} flattened features, "
            f"got {X_val.shape[1]}"
        )

    if X_test.shape[1] != expected_input_features:
        raise ValueError(
            f"Expected {expected_input_features} flattened features, "
            f"got {X_test.shape[1]}"
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
    # CHECK MASKS
    # ========================================================

    print()
    print("[3/6] Checking target masks...")

    train_valid = int(
        np.sum(Y_mask_train)
    )

    val_valid = int(
        np.sum(Y_mask_val)
    )

    test_valid = int(
        np.sum(Y_mask_test)
    )

    train_total = Y_mask_train.size
    val_total = Y_mask_val.size
    test_total = Y_mask_test.size

    print(
        f"  Train valid targets: "
        f"{train_valid}/{train_total}"
    )

    print(
        f"  Val valid targets  : "
        f"{val_valid}/{val_total}"
    )

    print(
        f"  Test valid targets : "
        f"{test_valid}/{test_total}"
    )

    # --------------------------------------------------------
    # XGBoost cannot directly train on a different missing
    # target mask for each output. For each output we therefore
    # select only rows whose corresponding target is valid.
    # --------------------------------------------------------

    output_masks_train = Y_mask_train.reshape(
        Y_mask_train.shape[0],
        OUTPUT_SIZE,
    )

    output_masks_val = Y_mask_val.reshape(
        Y_mask_val.shape[0],
        OUTPUT_SIZE,
    )

    # ========================================================
    # TRAIN 12 REGRESSORS
    # ========================================================

    print()
    print("[4/6] Training 12 XGBoost regressors...")

    print()
    print(
        "One regressor is trained for each "
        "horizon × target combination:"
    )

    print(
        "  H+1 O3, H+1 NO2, ..., H+6 O3, H+6 NO2"
    )

    models = []

    train_start = time.time()

    for output_idx in range(OUTPUT_SIZE):

        horizon = (
            output_idx // NUM_TARGETS
        ) + 1

        target_index = (
            output_idx % NUM_TARGETS
        )

        target_name = TARGET_NAMES[
            target_index
        ]

        valid_rows = (
            output_masks_train[:, output_idx]
            .astype(bool)
        )

        if not np.any(valid_rows):
            raise ValueError(
                f"No valid training targets for "
                f"H+{horizon} {target_name}"
            )

        model = build_model()

        print()
        print(
            f"[{output_idx + 1:02d}/{OUTPUT_SIZE}] "
            f"Training H+{horizon} {target_name} "
            f"({np.sum(valid_rows)} samples)"
        )

        model.fit(
            X_train[valid_rows],
            Y_train_flat[
                valid_rows,
                output_idx,
            ],
            verbose=False,
        )

        models.append(model)

    train_seconds = (
        time.time() - train_start
    )

    print()
    print(
        f"XGBoost training complete in "
        f"{train_seconds:.2f} seconds."
    )

    # ========================================================
    # VALIDATION / TEST PREDICTIONS
    # ========================================================

    print()
    print("[5/6] Generating validation and test predictions...")

    val_pred_flat = np.zeros(
        (
            X_val.shape[0],
            OUTPUT_SIZE,
        ),
        dtype=np.float32,
    )

    test_pred_flat = np.zeros(
        (
            X_test.shape[0],
            OUTPUT_SIZE,
        ),
        dtype=np.float32,
    )

    for output_idx, model in enumerate(models):

        val_pred_flat[:, output_idx] = (
            model.predict(X_val)
        )

        test_pred_flat[:, output_idx] = (
            model.predict(X_test)
        )

    val_pred_scaled = unflatten_predictions(
        val_pred_flat
    )

    test_pred_scaled = unflatten_predictions(
        test_pred_flat
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
    # INVERSE SCALE
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
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("XGBOOST — FINAL TEST RESULTS")
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
    # SAVE MODELS
    # ========================================================

    print()
    print("Saving trained models...")

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for idx, model in enumerate(models):

        horizon = (
            idx // NUM_TARGETS
        ) + 1

        target_name = TARGET_NAMES[
            idx % NUM_TARGETS
        ]

        model_path = (
            model_dir
            / f"xgboost_h{horizon}_{target_name}.joblib"
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
        "model": "xgboost",
        "model_type": "XGBRegressor",
        "target_mode": "direct_target_prediction",
        "research_contract": {
            "features": NUM_FEATURES,
            "tin": TIN,
            "tout": TOUT,
            "targets": TARGET_NAMES,
            "flattened_input_features": (
                TIN * NUM_FEATURES
            ),
            "flattened_outputs": OUTPUT_SIZE,
            "residual_learning": False,
        },
        "dataset": {
            "artifact": splits["data_path"],
            "train_samples": int(
                X_train.shape[0]
            ),
            "val_samples": int(
                X_val.shape[0]
            ),
            "test_samples": int(
                X_test.shape[0]
            ),
        },
        "configuration": {
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
            "tree_method": TREE_METHOD,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "training_seconds": train_seconds,
        "total_runtime_seconds": (
            time.time() - total_start
        ),
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
        "model": "xgboost",
        "model_type": "XGBRegressor",
        "number_of_regressors": OUTPUT_SIZE,
        "direct_target_prediction": True,
        "residual_learning": False,
        "dataset_artifact": splits["data_path"],
        "feature_schema": splits["feature_list_path"],
        "target_scaler": splits["target_scaler_path"],
        "contract": {
            "feature_count": NUM_FEATURES,
            "input_hours": TIN,
            "forecast_hours": TOUT,
            "target_count": NUM_TARGETS,
            "targets": TARGET_NAMES,
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
    print("XGBOOST COMPLETE")
    print("=" * 70)

    print()
    print("Saved:")
    print(
        f"  Models      : {model_dir}"
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
        f"{time.time() - total_start:.2f} seconds"
    )

    print()
    print("RESEARCH CONTRACT: PASSED")
    print("XGBOOST: COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()