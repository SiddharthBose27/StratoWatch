"""
StratoWatch 2.0 — Final Single-Site Transformer Evaluation

Evaluation-only script for the frozen final Transformer.

Contract:
    Input  : (N, 24, 134)
    Output : (N, 6, 2)

Uses:
    - Official Phase 7 test artifact
    - Frozen best_transformer.pt
    - Official target scaler
    - Official test masks
    - Real-unit metrics

NO TRAINING IS PERFORMED.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.temporal_transformer import TemporalTransformer


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
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

TARGET_SCALER_PATH = (
    ARTIFACT_DIR
    / "stratowatch_single_v2_target_scaler.pkl"
)

FEATURE_LIST_PATH = (
    PROJECT_DIR
    / "data"
    / "feature_list.json"
)

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "final_single_site"
    / "best_transformer.pt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "final_single_site"
)


# ============================================================
# 2. FROZEN MODEL CONTRACT
# ============================================================

SEED = 42

TIN = 24
TOUT = 6

NUM_FEATURES = 134
NUM_TARGETS = 2

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
DROPOUT = 0.1

BATCH_SIZE = 64

TARGET_NAMES = [
    "O3_target",
    "NO2_target",
]


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 5. METRICS
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

        valid = mask[
            :,
            :,
            target_idx,
        ].astype(bool)

        truth_target = truth[
            :,
            :,
            target_idx,
        ][valid]

        predictions_target = predictions[
            :,
            :,
            target_idx,
        ][valid]

        results[target_name] = calculate_metrics(
            truth[
                :,
                :,
                target_idx,
            ],
            predictions[
                :,
                :,
                target_idx,
            ],
            mask[
                :,
                :,
                target_idx,
            ],
        )

    return results


def calculate_horizon_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict:

    results = {}

    for h in range(truth.shape[1]):

        results[f"horizon_{h + 1}"] = calculate_metrics(
            truth[:, h, :],
            predictions[:, h, :],
            mask[:, h, :],
        )

    return results


# ============================================================
# 6. INVERSE TRANSFORM
# ============================================================

def inverse_transform_targets(
    values_scaled: np.ndarray,
    scaler,
) -> np.ndarray:

    original_shape = values_scaled.shape

    flattened = values_scaled.reshape(
        -1,
        NUM_TARGETS,
    )

    restored = scaler.inverse_transform(
        flattened
    )

    return restored.reshape(
        original_shape
    )


# ============================================================
# 7. LOAD DATA
# ============================================================

print("=" * 72)
print("STRATOWATCH 2.0 — FINAL TRANSFORMER EVALUATION")
print("=" * 72)

print(
    f"Device: {DEVICE}"
)

print(
    "\n[1/7] Loading official Phase 7 test artifact..."
)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Phase 7 artifact not found:\n{DATA_PATH}"
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
# 8. VALIDATE CONTRACT
# ============================================================

print(
    "\n[2/7] Validating research contract..."
)

assert X_test.shape[1:] == (
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
    f"  ✓ Input: {TIN} hours × {NUM_FEATURES} features"
)

print(
    f"  ✓ Output: {TOUT} hours × {NUM_TARGETS} targets"
)

print(
    "  ✓ Finite-value checks passed"
)

print(
    "  ✓ Test mask aligned"
)


# ============================================================
# 9. LOAD FEATURE CONTRACT
# ============================================================

print(
    "\n[3/7] Loading authoritative feature schema..."
)

if not FEATURE_LIST_PATH.exists():
    raise FileNotFoundError(
        f"Feature list not found:\n{FEATURE_LIST_PATH}"
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

o3_forecast_idx = feature_names.index(
    "O3_forecast"
)

no2_forecast_idx = feature_names.index(
    "NO2_forecast"
)

print(
    f"  ✓ Feature count: {len(feature_names)}"
)

print(
    f"  ✓ O3_forecast index: {o3_forecast_idx}"
)

print(
    f"  ✓ NO2_forecast index: {no2_forecast_idx}"
)


# ============================================================
# 10. LOAD TARGET SCALER
# ============================================================

print(
    "\n[4/7] Loading target scaler..."
)

if not TARGET_SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Target scaler not found:\n{TARGET_SCALER_PATH}"
    )

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)

print(
    f"  ✓ {TARGET_SCALER_PATH.name}"
)


# ============================================================
# 11. BUILD EXACT FROZEN MODEL
# ============================================================

print(
    "\n[5/7] Loading frozen Transformer..."
)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Transformer checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )

model = TemporalTransformer(
    in_dim=NUM_FEATURES,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    out_dim=NUM_TARGETS,
    tout=TOUT,
    dropout=DROPOUT,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

# Handle both a raw state_dict and a metadata-bearing checkpoint.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True,
)

model.eval()

print(
    "  ✓ Checkpoint loaded successfully"
)

print(
    f"  ✓ Input features: {NUM_FEATURES}"
)

print(
    f"  ✓ d_model: {D_MODEL}"
)

print(
    f"  ✓ attention heads: {NHEAD}"
)

print(
    f"  ✓ Transformer layers: {NUM_LAYERS}"
)

print(
    f"  ✓ Output horizon: {TOUT}"
)

print(
    "  ✓ Strict state-dict compatibility passed"
)


# ============================================================
# 12. GENERATE TEST PREDICTIONS
# ============================================================

print(
    "\n[6/7] Generating test predictions..."
)

predictions_scaled = []

with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_test),
        )

        X_batch = torch.from_numpy(
            X_test[start:end]
        ).to(DEVICE)

        output = model(
            X_batch
        )

        # Defensive handling in case the model
        # returns auxiliary outputs.
        if isinstance(output, tuple):
            output = output[0]

        if output.ndim != 3:
            raise ValueError(
                "Unexpected Transformer output shape: "
                f"{tuple(output.shape)}"
            )

        expected_shape = (
            end - start,
            TOUT,
            NUM_TARGETS,
        )

        if tuple(output.shape) != expected_shape:
            raise ValueError(
                "Transformer output contract failed. "
                f"Expected {expected_shape}, "
                f"got {tuple(output.shape)}"
            )

        predictions_scaled.append(
            output.detach()
            .cpu()
            .numpy()
        )

predictions_scaled = np.concatenate(
    predictions_scaled,
    axis=0,
)

if not np.isfinite(
    predictions_scaled
).all():

    raise FloatingPointError(
        "Non-finite Transformer predictions."
    )

print(
    "  Predictions:",
    predictions_scaled.shape
)

print(
    "  ✓ Prediction contract passed"
)


# ============================================================
# 13. CONVERT TO REAL UNITS
# ============================================================

print(
    "\n[7/7] Inverse-transforming targets..."
)

truth_real = inverse_transform_targets(
    Y_test,
    target_scaler,
)

predictions_real = inverse_transform_targets(
    predictions_scaled,
    target_scaler,
)

print(
    "  ✓ Predictions converted to real units"
)


# ============================================================
# 14. CALCULATE METRICS
# ============================================================

overall_metrics = calculate_metrics(
    truth_real,
    predictions_real,
    Y_mask_test,
)

target_metrics = calculate_target_metrics(
    truth_real,
    predictions_real,
    Y_mask_test,
)

horizon_metrics = calculate_horizon_metrics(
    truth_real,
    predictions_real,
    Y_mask_test,
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print()
print("=" * 72)
print("TRANSFORMER — FINAL TEST RESULTS (REAL UNITS)")
print("=" * 72)

print(
    "\nOVERALL"
)

print(
    f"MAE : {overall_metrics['MAE']:.4f}"
)

print(
    f"RMSE: {overall_metrics['RMSE']:.4f}"
)

print(
    f"R²  : {overall_metrics['R2']:.4f}"
)

print(
    "\nO3"
)

print(
    f"MAE : {target_metrics['O3_target']['MAE']:.4f}"
)

print(
    f"RMSE: {target_metrics['O3_target']['RMSE']:.4f}"
)

print(
    f"R²  : {target_metrics['O3_target']['R2']:.4f}"
)

print(
    "\nNO2"
)

print(
    f"MAE : {target_metrics['NO2_target']['MAE']:.4f}"
)

print(
    f"RMSE: {target_metrics['NO2_target']['RMSE']:.4f}"
)

print(
    f"R²  : {target_metrics['NO2_target']['R2']:.4f}"
)

print(
    "\nHORIZON-WISE"
)

for h in range(TOUT):

    metrics = horizon_metrics[
        f"horizon_{h + 1}"
    ]

    print(
        f"H+{h + 1} | "
        f"MAE={metrics['MAE']:.4f} | "
        f"RMSE={metrics['RMSE']:.4f} | "
        f"R²={metrics['R2']:.4f}"
    )


# ============================================================
# 16. SAVE EVALUATION ARTIFACTS
# ============================================================

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
    "model": "TemporalTransformer",
    "evaluation_type": "final_test_real_units",
    "seed": SEED,
    "device": str(DEVICE),
    "num_features": NUM_FEATURES,
    "tin": TIN,
    "tout": TOUT,
    "num_targets": NUM_TARGETS,
    "targets": TARGET_NAMES,
    "checkpoint": str(CHECKPOINT_PATH),
    "dataset": str(DATA_PATH),
    "target_scaler": str(TARGET_SCALER_PATH),
    "overall": overall_metrics,
    "per_target": target_metrics,
    "horizon_wise": horizon_metrics,
}

with open(
    OUTPUT_DIR / "final_evaluation_metrics.json",
    "w",
) as f:
    json.dump(
        metrics_payload,
        f,
        indent=2,
    )


# ============================================================
# 17. FINAL VALIDATION
# ============================================================

assert predictions_real.shape == Y_test.shape
assert truth_real.shape == Y_test.shape
assert Y_mask_test.shape == Y_test.shape

assert np.isfinite(
    predictions_real
).all()

assert np.isfinite(
    truth_real
).all()

print()
print("=" * 72)
print("TRANSFORMER EVALUATION COMPLETE")
print("=" * 72)

print(
    "\nSaved:"
)

print(
    f"  Predictions : "
    f"{OUTPUT_DIR / 'test_predictions_real_units.npy'}"
)

print(
    f"  Truth       : "
    f"{OUTPUT_DIR / 'test_truth_real_units.npy'}"
)

print(
    f"  Mask        : "
    f"{OUTPUT_DIR / 'test_mask.npy'}"
)

print(
    f"  Metrics     : "
    f"{OUTPUT_DIR / 'final_evaluation_metrics.json'}"
)

print(
    "\nRESEARCH CONTRACT: PASSED"
)