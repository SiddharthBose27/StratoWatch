"""
StratoWatch 2.0 — Forecast-Only Baseline

Definition:
    At the beginning of the 6-hour forecast horizon, use the
    latest available O3_forecast and NO2_forecast values from
    the 24-hour input window and persist them across H+1...H+6.

This baseline uses no learned model and no future information.

Evaluation:
    - Official Phase 7 test artifact
    - Same test masks as all final models
    - Real-unit MAE / RMSE / R²
    - O3 and NO2 metrics
    - Horizon-wise metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent

ARTIFACT_DIR = (
    REPO_ROOT
    / "backend"
    / "stratowatch_data"
    / "artifacts"
    / "test_singlesite"
)

DATA_PATH = (
    ARTIFACT_DIR
    / "stratowatch_single_v2_scaled.npz"
)

FEATURE_SCALER_PATH = (
    ARTIFACT_DIR
    / "stratowatch_single_v2_feature_scaler.pkl"
)

TARGET_SCALER_PATH = (
    ARTIFACT_DIR
    / "stratowatch_single_v2_target_scaler.pkl"
)

FEATURE_LIST_PATH = (
    PROJECT_DIR
    / "data"
    / "feature_list.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "final_baselines"
    / "forecast_only"
)


# ============================================================
# CONTRACT
# ============================================================

TIN = 24
TOUT = 6
NUM_FEATURES = 134
NUM_TARGETS = 2

TARGET_NAMES = [
    "O3_target",
    "NO2_target",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict:

    valid = mask.astype(bool)

    truth_flat = truth[valid]
    predictions_flat = predictions[valid]

    if len(truth_flat) == 0:
        return {
            "MAE": None,
            "RMSE": None,
            "R2": None,
        }

    mae = mean_absolute_error(
        truth_flat,
        predictions_flat,
    )

    rmse = np.sqrt(
        mean_squared_error(
            truth_flat,
            predictions_flat,
        )
    )

    r2 = r2_score(
        truth_flat,
        predictions_flat,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def calculate_target_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict:

    results = {}

    for target_idx, target_name in enumerate(
        TARGET_NAMES
    ):

        results[target_name] = calculate_metrics(
            truth[:, :, target_idx],
            predictions[:, :, target_idx],
            mask[:, :, target_idx],
        )

    return results


def calculate_horizon_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict:

    results = {}

    for h in range(TOUT):

        results[f"horizon_{h + 1}"] = calculate_metrics(
            truth[:, h, :],
            predictions[:, h, :],
            mask[:, h, :],
        )

    return results


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("STRATOWATCH 2.0 — FORECAST-ONLY BASELINE")
print("=" * 72)

print(
    "\nDefinition:"
)

print(
    "  Latest available O3/NO2 forecast is persisted "
    "across H+1...H+6."
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n[1/6] Loading official Phase 7 test artifact..."
)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Artifact not found:\n{DATA_PATH}"
    )

data = np.load(
    DATA_PATH,
    allow_pickle=False,
)

X_test = data[
    "X_test"
].astype(np.float32)

Y_test = data[
    "Y_test"
].astype(np.float32)

Y_mask_test = data[
    "Y_mask_test"
].astype(bool)

print(
    "  X_test:",
    X_test.shape
)

print(
    "  Y_test:",
    Y_test.shape
)

print(
    "  Y_mask_test:",
    Y_mask_test.shape
)


# ============================================================
# VALIDATE CONTRACT
# ============================================================

assert X_test.shape == (
    X_test.shape[0],
    TIN,
    NUM_FEATURES,
)

assert Y_test.shape[1:] == (
    TOUT,
    NUM_TARGETS,
)

assert Y_mask_test.shape == Y_test.shape

assert np.isfinite(X_test).all()
assert np.isfinite(Y_test).all()

print(
    "  ✓ 134 features"
)

print(
    "  ✓ 24-hour input"
)

print(
    "  ✓ 6-hour target horizon"
)

print(
    "  ✓ O3 + NO2 targets"
)


# ============================================================
# FEATURE SCHEMA
# ============================================================

print(
    "\n[2/6] Loading authoritative feature schema..."
)

with open(
    FEATURE_LIST_PATH,
    "r",
) as f:
    feature_names = json.load(f)

if len(feature_names) != NUM_FEATURES:
    raise ValueError(
        f"Expected {NUM_FEATURES} features, "
        f"found {len(feature_names)}."
    )

o3_idx = feature_names.index(
    "O3_forecast"
)

no2_idx = feature_names.index(
    "NO2_forecast"
)

print(
    f"  ✓ O3_forecast index: {o3_idx}"
)

print(
    f"  ✓ NO2_forecast index: {no2_idx}"
)


# ============================================================
# LOAD SCALERS
# ============================================================

print(
    "\n[3/6] Loading scalers..."
)

feature_scaler = joblib.load(
    FEATURE_SCALER_PATH
)

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)

print(
    "  ✓ Feature scaler loaded"
)

print(
    "  ✓ Target scaler loaded"
)


# ============================================================
# RECOVER LATEST FORECAST VALUES
# ============================================================

print(
    "\n[4/6] Constructing forecast-only predictions..."
)

# Phase 7 X is scaled.
#
# Recover only the two forecast features from the feature scaler.
#
# For StandardScaler:
#     x_real = x_scaled * scale + mean
#
# We deliberately use only the LAST available timestep,
# which is the final timestamp of the 24-hour input window.

latest_scaled = X_test[
    :,
    -1,
    :
]

if not hasattr(
    feature_scaler,
    "mean_"
) or not hasattr(
    feature_scaler,
    "scale_"
):
    raise TypeError(
        "Expected a StandardScaler-like feature scaler "
        "with mean_ and scale_."
    )

latest_o3_forecast = (
    latest_scaled[:, o3_idx]
    * feature_scaler.scale_[o3_idx]
    + feature_scaler.mean_[o3_idx]
)

latest_no2_forecast = (
    latest_scaled[:, no2_idx]
    * feature_scaler.scale_[no2_idx]
    + feature_scaler.mean_[no2_idx]
)

# Repeat the latest available forecast through all six
# forecast horizons.

predictions_real = np.zeros(
    (
        len(X_test),
        TOUT,
        NUM_TARGETS,
    ),
    dtype=np.float64,
)

predictions_real[
    :,
    :,
    0
] = latest_o3_forecast[:, None]

predictions_real[
    :,
    :,
    1
] = latest_no2_forecast[:, None]

# Y_test is scaled, so convert the ground truth to real units.

truth_real = target_scaler.inverse_transform(
    Y_test.reshape(
        -1,
        NUM_TARGETS,
    )
).reshape(
    Y_test.shape
)

if not np.isfinite(
    predictions_real
).all():
    raise FloatingPointError(
        "Non-finite forecast baseline predictions."
    )

print(
    "  ✓ Latest forecast values recovered"
)

print(
    "  ✓ Forecast persisted across 6 horizons"
)

print(
    "  ✓ No future information used"
)


# ============================================================
# METRICS
# ============================================================

print(
    "\n[5/6] Calculating real-unit metrics..."
)

overall = calculate_metrics(
    truth_real,
    predictions_real,
    Y_mask_test,
)

per_target = calculate_target_metrics(
    truth_real,
    predictions_real,
    Y_mask_test,
)

horizon_wise = calculate_horizon_metrics(
    truth_real,
    predictions_real,
    Y_mask_test,
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 72)
print("FORECAST-ONLY BASELINE — FINAL TEST RESULTS")
print("=" * 72)

print(
    "\nOVERALL"
)

print(
    f"MAE : {overall['MAE']:.4f}"
)

print(
    f"RMSE: {overall['RMSE']:.4f}"
)

print(
    f"R²  : {overall['R2']:.4f}"
)

print(
    "\nO3"
)

print(
    f"MAE : {per_target['O3_target']['MAE']:.4f}"
)

print(
    f"RMSE: {per_target['O3_target']['RMSE']:.4f}"
)

print(
    f"R²  : {per_target['O3_target']['R2']:.4f}"
)

print(
    "\nNO2"
)

print(
    f"MAE : {per_target['NO2_target']['MAE']:.4f}"
)

print(
    f"RMSE: {per_target['NO2_target']['RMSE']:.4f}"
)

print(
    f"R²  : {per_target['NO2_target']['R2']:.4f}"
)

print(
    "\nHORIZON-WISE"
)

for h in range(TOUT):

    metrics = horizon_wise[
        f"horizon_{h + 1}"
    ]

    print(
        f"H+{h + 1} | "
        f"MAE={metrics['MAE']:.4f} | "
        f"RMSE={metrics['RMSE']:.4f} | "
        f"R²={metrics['R2']:.4f}"
    )


# ============================================================
# SAVE ARTIFACTS
# ============================================================

print(
    "\n[6/6] Saving research artifacts..."
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(
    OUTPUT_DIR / "test_predictions_real_units.npy",
    predictions_real,
)

np.save(
    OUTPUT_DIR / "test_truth_real_units.npy",
    truth_real,
)

np.save(
    OUTPUT_DIR / "test_mask.npy",
    Y_mask_test,
)

metrics_payload = {
    "model": "forecast_only",
    "definition": (
        "Persist the latest available O3_forecast and "
        "NO2_forecast values from the final timestep of "
        "the 24-hour input window across H+1...H+6."
    ),
    "tin": TIN,
    "tout": TOUT,
    "num_features": NUM_FEATURES,
    "targets": TARGET_NAMES,
    "dataset": str(DATA_PATH),
    "feature_scaler": str(FEATURE_SCALER_PATH),
    "target_scaler": str(TARGET_SCALER_PATH),
    "forecast_feature_indices": {
        "O3_forecast": o3_idx,
        "NO2_forecast": no2_idx,
    },
    "overall": overall,
    "per_target": per_target,
    "horizon_wise": horizon_wise,
}

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
) as f:
    json.dump(
        metrics_payload,
        f,
        indent=2,
    )

print(
    "  ✓ Predictions saved"
)

print(
    "  ✓ Ground truth saved"
)

print(
    "  ✓ Mask saved"
)

print(
    "  ✓ Metrics saved"
)

print()
print("=" * 72)
print("FORECAST-ONLY BASELINE COMPLETE")
print("=" * 72)
print("RESEARCH CONTRACT: PASSED")