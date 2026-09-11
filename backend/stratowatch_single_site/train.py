"""
StratoWatch 2.0 — Final Single-Site Transformer

Direct multi-horizon forecasting:
    Input  : (N, 24, 134)
    Output : (N, 6, 2)

Targets:
    O3_target
    NO2_target

The Phase 7 artifact contains DIRECT SCALED TARGETS.
This trainer therefore does NOT perform residual
reconstruction.

CPU is intentionally used for the final run because
the previous MPS run produced intermittent inf losses.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader, TensorDataset

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

FEATURE_SCALER_PATH = (
    ARTIFACT_DIR
    / "stratowatch_single_v2_feature_scaler.pkl"
)

FEATURE_LIST_PATH = (
    PROJECT_DIR
    / "data"
    / "feature_list.json"
)

CONFIG_PATH = (
    ARTIFACT_DIR
    / "stratowatch_single_v2_config.json"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "stratowatch_single_v2_scaled_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "final_single_site"
)

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "best_transformer.pt"
)


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

TIN = 24
TOUT = 6

NUM_FEATURES = 134
NUM_TARGETS = 2

BATCH_SIZE = 64
MAX_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 7

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
DROPOUT = 0.1


# ============================================================
# 3. DEVICE
# ============================================================

# CPU is intentional.
#
# The previous MPS run produced:
#     Train MSE: inf
#
# We do not want an unstable checkpoint for the final
# research result.

DEVICE = torch.device("cpu")


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 5. METRIC FUNCTIONS
# ============================================================

def masked_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate global masked MSE.
    """

    valid = mask.bool()

    if not torch.isfinite(predictions).all():
        raise FloatingPointError(
            "Non-finite predictions encountered."
        )

    if not torch.isfinite(targets).all():
        raise FloatingPointError(
            "Non-finite targets encountered."
        )

    if not valid.any():
        raise ValueError(
            "No valid target values in batch."
        )

    error = predictions[valid] - targets[valid]

    loss = torch.mean(error * error)

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Non-finite loss encountered."
        )

    return loss


def calculate_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Calculate global masked MAE, RMSE and R².
    """

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


def calculate_per_target_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    target_names: list[str],
) -> dict:
    """
    Calculate metrics separately for O3 and NO2.
    """

    results = {}

    for target_idx, target_name in enumerate(
        target_names
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

        if len(truth_target) == 0:
            results[target_name] = {
                "MAE": None,
                "RMSE": None,
                "R2": None,
            }
            continue

        mae = mean_absolute_error(
            truth_target,
            predictions_target,
        )

        rmse = np.sqrt(
            mean_squared_error(
                truth_target,
                predictions_target,
            )
        )

        r2 = r2_score(
            truth_target,
            predictions_target,
        )

        results[target_name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

    return results


def calculate_horizon_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Calculate metrics for each forecast horizon.

    horizon_1 = first predicted hour
    horizon_6 = sixth predicted hour
    """

    results = {}

    for h in range(
        truth.shape[1]
    ):

        valid = mask[
            :,
            h,
            :,
        ].astype(bool)

        truth_h = truth[
            :,
            h,
            :,
        ][valid]

        predictions_h = predictions[
            :,
            h,
            :,
        ][valid]

        if len(truth_h) == 0:
            results[
                f"horizon_{h + 1}"
            ] = {
                "MAE": None,
                "RMSE": None,
                "R2": None,
            }
            continue

        mae = mean_absolute_error(
            truth_h,
            predictions_h,
        )

        rmse = np.sqrt(
            mean_squared_error(
                truth_h,
                predictions_h,
            )
        )

        r2 = r2_score(
            truth_h,
            predictions_h,
        )

        results[
            f"horizon_{h + 1}"
        ] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

    return results


def inverse_transform_targets(
    values_scaled: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Inverse-transform:
        (N, 6, 2)
    """

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
# 6. START
# ============================================================

print("=" * 72)
print(
    "STRATOWATCH 2.0 — FINAL SINGLE-SITE TRANSFORMER"
)
print("=" * 72)

print(
    f"Device: {DEVICE}"
)

print(
    f"Seed  : {SEED}"
)


# ============================================================
# 7. LOAD PHASE 7 ARTIFACT
# ============================================================

print(
    "\n[1/8] Phase 7 artifact loaded"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Phase 7 artifact not found:\n{DATA_PATH}"
    )

data = np.load(
    DATA_PATH,
    allow_pickle=False,
)

X_train = data[
    "X_train"
].astype(np.float32)

Y_train = data[
    "Y_train"
].astype(np.float32)

X_val = data[
    "X_val"
].astype(np.float32)

Y_val = data[
    "Y_val"
].astype(np.float32)

X_test = data[
    "X_test"
].astype(np.float32)

Y_test = data[
    "Y_test"
].astype(np.float32)

X_mask_train = data[
    "X_mask_train"
].astype(bool)

Y_mask_train = data[
    "Y_mask_train"
].astype(bool)

X_mask_val = data[
    "X_mask_val"
].astype(bool)

Y_mask_val = data[
    "Y_mask_val"
].astype(bool)

X_mask_test = data[
    "X_mask_test"
].astype(bool)

Y_mask_test = data[
    "Y_mask_test"
].astype(bool)


print(
    "  X_train:",
    X_train.shape
)

print(
    "  Y_train:",
    Y_train.shape
)

print(
    "  X_val  :",
    X_val.shape
)

print(
    "  Y_val  :",
    Y_val.shape
)

print(
    "  X_test :",
    X_test.shape
)

print(
    "  Y_test :",
    Y_test.shape
)


# ============================================================
# 8. VALIDATE DATA CONTRACT
# ============================================================

print(
    "\n[2/8] Masks loaded"
)

print(
    "  Y_train valid:",
    int(Y_mask_train.sum()),
    "/",
    Y_mask_train.size,
)

print(
    "  Y_val valid  :",
    int(Y_mask_val.sum()),
    "/",
    Y_mask_val.size,
)

print(
    "  Y_test valid :",
    int(Y_mask_test.sum()),
    "/",
    Y_mask_test.size,
)


expected_x_shape = (
    TIN,
    NUM_FEATURES,
)

expected_y_shape = (
    TOUT,
    NUM_TARGETS,
)

assert X_train.shape[1:] == expected_x_shape
assert X_val.shape[1:] == expected_x_shape
assert X_test.shape[1:] == expected_x_shape

assert Y_train.shape[1:] == expected_y_shape
assert Y_val.shape[1:] == expected_y_shape
assert Y_test.shape[1:] == expected_y_shape

assert X_mask_train.shape == X_train.shape
assert X_mask_val.shape == X_val.shape
assert X_mask_test.shape == X_test.shape

assert Y_mask_train.shape == Y_train.shape
assert Y_mask_val.shape == Y_val.shape
assert Y_mask_test.shape == Y_test.shape

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()

assert np.isfinite(Y_train).all()
assert np.isfinite(Y_val).all()
assert np.isfinite(Y_test).all()


print(
    f"  ✓ {NUM_FEATURES} input features"
)

print(
    f"  ✓ {TIN}-hour input window"
)

print(
    f"  ✓ {TOUT}-hour forecast horizon"
)

print(
    "  ✓ 2 targets: O3 and NO2"
)

print(
    "  ✓ Finite-value checks passed"
)

print(
    "  ✓ Phase 7 contract validated"
)


# ============================================================
# 9. LOAD AUTHORITATIVE FEATURE LIST
# ============================================================

if not FEATURE_LIST_PATH.exists():
    raise FileNotFoundError(
        f"Feature list not found:\n"
        f"{FEATURE_LIST_PATH}"
    )

with open(
    FEATURE_LIST_PATH,
    "r",
) as f:
    FEATURE_NAMES = json.load(f)

if len(FEATURE_NAMES) != NUM_FEATURES:
    raise ValueError(
        f"Expected {NUM_FEATURES} features, "
        f"found {len(FEATURE_NAMES)}."
    )

O3_FORECAST_IDX = FEATURE_NAMES.index(
    "O3_forecast"
)

NO2_FORECAST_IDX = FEATURE_NAMES.index(
    "NO2_forecast"
)

print(
    f"  ✓ O3_forecast index: "
    f"{O3_FORECAST_IDX}"
)

print(
    f"  ✓ NO2_forecast index: "
    f"{NO2_FORECAST_IDX}"
)


# ============================================================
# 10. LOAD SCALERS
# ============================================================

if not TARGET_SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Target scaler not found:\n"
        f"{TARGET_SCALER_PATH}"
    )

if not FEATURE_SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Feature scaler not found:\n"
        f"{FEATURE_SCALER_PATH}"
    )

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)

feature_scaler = joblib.load(
    FEATURE_SCALER_PATH
)

print(
    "  ✓ Target scaler loaded"
)

print(
    "  ✓ Feature scaler loaded"
)


# ============================================================
# 11. DATALOADERS
# ============================================================

print(
    "\n[3/8] DataLoaders created"
)

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(Y_train),
    torch.from_numpy(
        Y_mask_train.astype(
            np.float32
        )
    ),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(Y_val),
    torch.from_numpy(
        Y_mask_val.astype(
            np.float32
        )
    ),
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(Y_test),
    torch.from_numpy(
        Y_mask_test.astype(
            np.float32
        )
    ),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)

print(
    "  Train batches:",
    len(train_loader)
)

print(
    "  Val batches  :",
    len(val_loader)
)

print(
    "  Test batches :",
    len(test_loader)
)


# ============================================================
# 12. CREATE MODEL
# ============================================================

print(
    "\n[4/8] Model created"
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

print(
    "  Input features :",
    NUM_FEATURES
)

print(
    "  Input window   :",
    TIN
)

print(
    "  Forecast horizon:",
    TOUT
)

print(
    "  d_model:",
    D_MODEL
)

print(
    "  attention heads:",
    NHEAD
)

print(
    "  Transformer layers:",
    NUM_LAYERS
)


# ============================================================
# 13. OPTIMIZER + SCHEDULER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# 14. TRAINING
# ============================================================

print(
    "\n[5/8] Training"
)

print(
    "-" * 72
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

history = {
    "train_mse": [],
    "val_mse": [],
    "learning_rate": [],
}

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0


for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_batches = 0

    for (
        X_batch,
        Y_batch,
        M_batch,
    ) in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        M_batch = M_batch.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            X_batch
        )

        loss = masked_mse(
            predictions,
            Y_batch,
            M_batch,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at "
                f"epoch {epoch}."
            )

        loss.backward()

        # Safety guard against exploding gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        # Check the model after the optimizer step.
        for parameter in model.parameters():

            if not torch.isfinite(
                parameter
            ).all():

                raise FloatingPointError(
                    f"Non-finite model parameter "
                    f"detected after epoch {epoch}."
                )

        train_loss_sum += float(
            loss.detach().cpu()
        )

        train_batches += 1

    train_loss = (
        train_loss_sum
        / train_batches
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_batches = 0

    with torch.no_grad():

        for (
            X_batch,
            Y_batch,
            M_batch,
        ) in val_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            Y_batch = Y_batch.to(
                DEVICE
            )

            M_batch = M_batch.to(
                DEVICE
            )

            predictions = model(
                X_batch
            )

            loss = masked_mse(
                predictions,
                Y_batch,
                M_batch,
            )

            val_loss_sum += float(
                loss.detach().cpu()
            )

            val_batches += 1

    val_loss = (
        val_loss_sum
        / val_batches
    )

    scheduler.step(
        val_loss
    )

    current_lr = optimizer.param_groups[
        0
    ][
        "lr"
    ]

    history[
        "train_mse"
    ].append(
        float(train_loss)
    )

    history[
        "val_mse"
    ].append(
        float(val_loss)
    )

    history[
        "learning_rate"
    ].append(
        float(current_lr)
    )

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f} | "
        f"LR: {current_lr:.2e}"
    )

    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict":
                model.state_dict(),

            "epoch":
                epoch,

            "best_val_mse":
                float(best_val_loss),

            "seed":
                SEED,

            "device":
                str(DEVICE),

            "model_type":
                "TemporalTransformer",

            "task":
                "single_site_direct_multi_horizon_forecasting",

            "input_features":
                NUM_FEATURES,

            "input_window":
                TIN,

            "forecast_horizon":
                TOUT,

            "d_model":
                D_MODEL,

            "nhead":
                NHEAD,

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

            "target_names": [
                "O3_target",
                "NO2_target",
            ],

            "feature_names":
                FEATURE_NAMES,

            "artifact":
                str(DATA_PATH),

            "artifact_config":
                str(CONFIG_PATH),

            "artifact_metadata":
                str(METADATA_PATH),
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH,
        )

        print(
            "  ✓ Best checkpoint saved"
        )

    else:

        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping after "
                f"{epoch} epochs."
            )

            break


# ============================================================
# 15. SAVE TRAINING HISTORY
# ============================================================

print(
    "\n[6/8] Saving training artifacts"
)

history[
    "best_epoch"
] = best_epoch

history[
    "best_val_mse"
] = float(
    best_val_loss
)

history_path = (
    OUTPUT_DIR
    / "training_history.json"
)

with open(
    history_path,
    "w",
) as f:

    json.dump(
        history,
        f,
        indent=2,
    )


# ============================================================
# 16. SAVE LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["train_mse"],
    label="Train MSE",
)

plt.plot(
    history["val_mse"],
    label="Validation MSE",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Masked MSE"
)

plt.title(
    "StratoWatch Single-Site Transformer"
)

plt.legend()

plt.tight_layout()

loss_curve_path = (
    OUTPUT_DIR
    / "loss_curve.png"
)

plt.savefig(
    loss_curve_path,
    dpi=200,
)

plt.close()


print(
    "  ✓ training_history.json"
)

print(
    "  ✓ loss_curve.png"
)


# ============================================================
# 17. LOAD BEST CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        "Best checkpoint was not created."
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


# ============================================================
# 18. TEST PREDICTIONS
# ============================================================

print(
    "\n[7/8] Evaluating test set"
)

predictions_scaled = []
truth_scaled = []
masks = []

with torch.no_grad():

    for (
        X_batch,
        Y_batch,
        M_batch,
    ) in test_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        predictions = model(
            X_batch
        )

        predictions_np = (
            predictions
            .cpu()
            .numpy()
        )

        if not np.isfinite(
            predictions_np
        ).all():

            raise FloatingPointError(
                "Non-finite test predictions."
            )

        predictions_scaled.append(
            predictions_np
        )

        truth_scaled.append(
            Y_batch.numpy()
        )

        masks.append(
            M_batch.numpy()
            .astype(bool)
        )


predictions_scaled = np.concatenate(
    predictions_scaled,
    axis=0,
)

truth_scaled = np.concatenate(
    truth_scaled,
    axis=0,
)

masks = np.concatenate(
    masks,
    axis=0,
).astype(bool)


# ============================================================
# 19. INVERSE TRANSFORM
# ============================================================

predictions_real = (
    inverse_transform_targets(
        predictions_scaled,
        target_scaler,
    )
)

truth_real = (
    inverse_transform_targets(
        truth_scaled,
        target_scaler,
    )
)


if not np.isfinite(
    predictions_real
).all():

    raise FloatingPointError(
        "Non-finite inverse-transformed predictions."
    )

if not np.isfinite(
    truth_real
).all():

    raise FloatingPointError(
        "Non-finite inverse-transformed truth."
    )


# ============================================================
# 20. MODEL METRICS
# ============================================================

overall_metrics = calculate_metrics(
    truth_real,
    predictions_real,
    masks,
)

per_target_metrics = (
    calculate_per_target_metrics(
        truth_real,
        predictions_real,
        masks,
        [
            "O3",
            "NO2",
        ],
    )
)

horizon_metrics = (
    calculate_horizon_metrics(
        truth_real,
        predictions_real,
        masks,
    )
)


# ============================================================
# 21. FORECAST-REFERENCE BASELINE
# ============================================================
#
# The Phase 7 artifact does not contain a separate future
# six-hour forecast-field tensor.
#
# Therefore we cannot honestly recreate the old
# "forecast-only residual baseline".
#
# We instead use:
#
#     final available input forecast
#     repeated across six horizons
#
# and explicitly label it as a reference baseline.
# ============================================================

print(
    "\nCalculating forecast-reference baseline..."
)

reference_scaled = X_test[
    :,
    -1,
    :
].copy()

reference_features_real = (
    feature_scaler.inverse_transform(
        reference_scaled
    )
)

reference_real = np.stack(
    [
        reference_features_real[
            :,
            O3_FORECAST_IDX,
        ],

        reference_features_real[
            :,
            NO2_FORECAST_IDX,
        ],
    ],
    axis=-1,
)

reference_real = np.repeat(
    reference_real[
        :,
        None,
        :
    ],
    TOUT,
    axis=1,
)

forecast_reference_metrics = (
    calculate_metrics(
        truth_real,
        reference_real,
        masks,
    )
)

forecast_reference_per_target = (
    calculate_per_target_metrics(
        truth_real,
        reference_real,
        masks,
        [
            "O3",
            "NO2",
        ],
    )
)

forecast_reference_horizon = (
    calculate_horizon_metrics(
        truth_real,
        reference_real,
        masks,
    )
)


# ============================================================
# 22. SAVE TEST ARRAYS
# ============================================================

predictions_path = (
    OUTPUT_DIR
    / "test_predictions.npy"
)

truth_path = (
    OUTPUT_DIR
    / "test_truth.npy"
)

mask_path = (
    OUTPUT_DIR
    / "test_mask.npy"
)

np.save(
    predictions_path,
    predictions_real,
)

np.save(
    truth_path,
    truth_real,
)

np.save(
    mask_path,
    masks,
)


# ============================================================
# 23. SAVE METRICS
# ============================================================

metrics = {

    "experiment": {

        "model":
            "TemporalTransformer",

        "task":
            "single_site_direct_multi_horizon_forecasting",

        "dataset_version":
            "stratowatch_single_v2",

        "artifact_version":
            "artifact_v1",

        "seed":
            SEED,

        "device":
            str(DEVICE),

        "input_window":
            TIN,

        "forecast_horizon":
            TOUT,

        "num_features":
            NUM_FEATURES,

        "num_targets":
            NUM_TARGETS,

        "d_model":
            D_MODEL,

        "nhead":
            NHEAD,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "best_epoch":
            best_epoch,

        "best_val_mse":
            float(best_val_loss),
    },

    "data": {

        "artifact_path":
            str(DATA_PATH),

        "config_path":
            str(CONFIG_PATH),

        "metadata_path":
            str(METADATA_PATH),

        "target_scaler_path":
            str(TARGET_SCALER_PATH),

        "feature_scaler_path":
            str(FEATURE_SCALER_PATH),

        "feature_list_path":
            str(FEATURE_LIST_PATH),

        "train_shape":
            list(X_train.shape),

        "val_shape":
            list(X_val.shape),

        "test_shape":
            list(X_test.shape),
    },

    "transformer": {

        "overall":
            overall_metrics,

        "per_target":
            per_target_metrics,

        "per_horizon":
            horizon_metrics,
    },

    "forecast_reference_baseline": {

        "description":
            (
                "Final available input-timestep "
                "O3/NO2 forecast values repeated "
                "across the six forecast horizons. "
                "This is a reference baseline and "
                "not a full future-horizon forecast-"
                "field baseline."
            ),

        "overall":
            forecast_reference_metrics,

        "per_target":
            forecast_reference_per_target,

        "per_horizon":
            forecast_reference_horizon,
    },
}


metrics_path = (
    OUTPUT_DIR
    / "test_metrics.json"
)

with open(
    metrics_path,
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# 24. FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "FINAL SINGLE-SITE TRANSFORMER RESULTS"
)

print(
    "=" * 72
)

print(
    f"Overall MAE : "
    f"{overall_metrics['MAE']:.4f}"
)

print(
    f"Overall RMSE: "
    f"{overall_metrics['RMSE']:.4f}"
)

print(
    f"Overall R²  : "
    f"{overall_metrics['R2']:.4f}"
)


print(
    "\nO3"
)

print(
    f"  MAE : "
    f"{per_target_metrics['O3']['MAE']:.4f}"
)

print(
    f"  RMSE: "
    f"{per_target_metrics['O3']['RMSE']:.4f}"
)

print(
    f"  R²  : "
    f"{per_target_metrics['O3']['R2']:.4f}"
)


print(
    "\nNO2"
)

print(
    f"  MAE : "
    f"{per_target_metrics['NO2']['MAE']:.4f}"
)

print(
    f"  RMSE: "
    f"{per_target_metrics['NO2']['RMSE']:.4f}"
)

print(
    f"  R²  : "
    f"{per_target_metrics['NO2']['R2']:.4f}"
)


print(
    "\nFORECAST-REFERENCE BASELINE"
)

print(
    f"  MAE : "
    f"{forecast_reference_metrics['MAE']:.4f}"
)

print(
    f"  RMSE: "
    f"{forecast_reference_metrics['RMSE']:.4f}"
)

print(
    f"  R²  : "
    f"{forecast_reference_metrics['R2']:.4f}"
)


# ============================================================
# 25. OUTPUT FILES
# ============================================================

print(
    "\nSaved:"
)

print(
    f"  {CHECKPOINT_PATH}"
)

print(
    f"  {history_path}"
)

print(
    f"  {loss_curve_path}"
)

print(
    f"  {predictions_path}"
)

print(
    f"  {truth_path}"
)

print(
    f"  {mask_path}"
)

print(
    f"  {metrics_path}"
)

print(
    "\n"
    + "=" * 72
)

print(
    "SINGLE-SITE TRANSFORMER TRAINING COMPLETE"
)

print(
    "=" * 72
)
