"""
Common data loader for the final StratoWatch single-site experiments.

Research contract:
    - Official Phase 7 artifact
    - 134 input features
    - 24-hour input window
    - 6-hour forecast horizon
    - O3 + NO2 targets
    - Same train/validation/test splits for every baseline
    - Same target scaler for real-unit evaluation
    - Same masks for all models

This file is intentionally shared by:
    - Forecast-only baseline
    - Random Forest
    - XGBoost
    - LSTM
    - TCN
    - Transformer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SINGLE_SITE_DIR = BASE_DIR.parent
BACKEND_DIR = SINGLE_SITE_DIR.parent

PHASE7_ARTIFACT_DIR = (
    BACKEND_DIR
    / "stratowatch_data"
    / "artifacts"
    / "test_singlesite"
)

DATA_PATH = (
    PHASE7_ARTIFACT_DIR
    / "stratowatch_single_v2_scaled.npz"
)

FEATURE_LIST_PATH = (
    SINGLE_SITE_DIR
    / "data"
    / "feature_list.json"
)

TARGET_SCALER_PATH = (
    PHASE7_ARTIFACT_DIR
    / "stratowatch_single_v2_target_scaler.pkl"
)

TARGET_SCALER_JSON_PATH = (
    PHASE7_ARTIFACT_DIR
    / "stratowatch_single_v2_target_scaler.json"
)

FEATURE_SCALER_PATH = (
    PHASE7_ARTIFACT_DIR
    / "stratowatch_single_v2_feature_scaler.pkl"
)

METADATA_PATH = (
    PHASE7_ARTIFACT_DIR
    / "stratowatch_single_v2_scaled_metadata.json"
)


# ============================================================
# EXPECTED RESEARCH CONTRACT
# ============================================================

EXPECTED_FEATURE_COUNT = 134
EXPECTED_TIN = 24
EXPECTED_TOUT = 6
EXPECTED_TARGET_COUNT = 2

TARGET_NAMES = [
    "O3_target",
    "NO2_target",
]

EXPECTED_SHAPES = {
    "X_train": (30438, 24, 134),
    "Y_train": (30438, 6, 2),
    "X_val": (6522, 24, 134),
    "Y_val": (6522, 6, 2),
    "X_test": (6464, 24, 134),
    "Y_test": (6464, 6, 2),
}


# ============================================================
# HELPERS
# ============================================================

def _require_file(path: Path, description: str) -> None:
    """Raise a clear error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def _load_feature_names() -> list[str]:
    """Load the authoritative 134-feature schema."""
    _require_file(
        FEATURE_LIST_PATH,
        "Authoritative feature list",
    )

    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    if not isinstance(feature_names, list):
        raise ValueError(
            f"Feature list must be a JSON list: {FEATURE_LIST_PATH}"
        )

    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Feature schema mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT} features\n"
            f"Found:    {len(feature_names)} features\n"
            f"Path:     {FEATURE_LIST_PATH}"
        )

    return feature_names


def _validate_array(
    name: str,
    array: np.ndarray,
    expected_shape: Tuple[int, ...],
) -> None:
    """Validate an artifact array against the final research contract."""

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Found:    {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )


def _load_target_scaler(path: Optional[Path] = None):
    """Load the official Phase 7 target scaler."""

    scaler_path = (
        Path(path)
        if path is not None
        else TARGET_SCALER_PATH
    )

    _require_file(
        scaler_path,
        "Phase 7 target scaler",
    )

    scaler = joblib.load(scaler_path)

    if not hasattr(scaler, "transform"):
        raise TypeError(
            f"Loaded target scaler does not provide transform():\n"
            f"{scaler_path}"
        )

    if not hasattr(scaler, "inverse_transform"):
        raise TypeError(
            f"Loaded target scaler does not provide inverse_transform():\n"
            f"{scaler_path}"
        )

    return scaler


# ============================================================
# MAIN DATA LOADER
# ============================================================

def load_splits(
    data_path: Optional[Path] = None,
    target_scaler_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load the official Phase 7 single-site dataset.

    Returns
    -------
    dict
        Contains sequence-format arrays, flattened arrays,
        masks, feature names, target names, and scaler metadata.
    """

    data_file = (
        Path(data_path)
        if data_path is not None
        else DATA_PATH
    )

    _require_file(
        data_file,
        "Phase 7 single-site NPZ artifact",
    )

    feature_names = _load_feature_names()

    target_scaler = _load_target_scaler(
        target_scaler_path
    )

    # --------------------------------------------------------
    # Load NPZ
    # --------------------------------------------------------

    data = np.load(data_file, allow_pickle=True)

    required_keys = [
        "X_train",
        "Y_train",
        "X_val",
        "Y_val",
        "X_test",
        "Y_test",
        "X_mask_train",
        "Y_mask_train",
        "X_mask_val",
        "Y_mask_val",
        "X_mask_test",
        "Y_mask_test",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in data
    ]

    if missing_keys:
        raise KeyError(
            "Phase 7 NPZ is missing required arrays:\n"
            + "\n".join(f"  - {key}" for key in missing_keys)
        )

    # --------------------------------------------------------
    # Load arrays
    # --------------------------------------------------------

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    Y_train = np.asarray(data["Y_train"], dtype=np.float32)

    X_val = np.asarray(data["X_val"], dtype=np.float32)
    Y_val = np.asarray(data["Y_val"], dtype=np.float32)

    X_test = np.asarray(data["X_test"], dtype=np.float32)
    Y_test = np.asarray(data["Y_test"], dtype=np.float32)

    X_mask_train = np.asarray(
        data["X_mask_train"],
        dtype=np.float32,
    )

    Y_mask_train = np.asarray(
        data["Y_mask_train"],
        dtype=np.float32,
    )

    X_mask_val = np.asarray(
        data["X_mask_val"],
        dtype=np.float32,
    )

    Y_mask_val = np.asarray(
        data["Y_mask_val"],
        dtype=np.float32,
    )

    X_mask_test = np.asarray(
        data["X_mask_test"],
        dtype=np.float32,
    )

    Y_mask_test = np.asarray(
        data["Y_mask_test"],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validate research shapes
    # --------------------------------------------------------

    _validate_array(
        "X_train",
        X_train,
        EXPECTED_SHAPES["X_train"],
    )

    _validate_array(
        "Y_train",
        Y_train,
        EXPECTED_SHAPES["Y_train"],
    )

    _validate_array(
        "X_val",
        X_val,
        EXPECTED_SHAPES["X_val"],
    )

    _validate_array(
        "Y_val",
        Y_val,
        EXPECTED_SHAPES["Y_val"],
    )

    _validate_array(
        "X_test",
        X_test,
        EXPECTED_SHAPES["X_test"],
    )

    _validate_array(
        "Y_test",
        Y_test,
        EXPECTED_SHAPES["Y_test"],
    )

    # --------------------------------------------------------
    # Validate masks
    # --------------------------------------------------------

    expected_mask_shapes = {
        "X_mask_train": X_train.shape,
        "Y_mask_train": Y_train.shape,
        "X_mask_val": X_val.shape,
        "Y_mask_val": Y_val.shape,
        "X_mask_test": X_test.shape,
        "Y_mask_test": Y_test.shape,
    }

    mask_arrays = {
        "X_mask_train": X_mask_train,
        "Y_mask_train": Y_mask_train,
        "X_mask_val": X_mask_val,
        "Y_mask_val": Y_mask_val,
        "X_mask_test": X_mask_test,
        "Y_mask_test": Y_mask_test,
    }

    for name, mask in mask_arrays.items():

        if mask.shape != expected_mask_shapes[name]:
            raise ValueError(
                f"{name} shape mismatch.\n"
                f"Expected: {expected_mask_shapes[name]}\n"
                f"Found:    {mask.shape}"
            )

        if not np.all(np.isfinite(mask)):
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        unique_values = np.unique(mask)

        if not np.all(
            np.isin(unique_values, [0.0, 1.0])
        ):
            raise ValueError(
                f"{name} must contain only 0/1 values.\n"
                f"Found values: {unique_values}"
            )

    # --------------------------------------------------------
    # Flattened representations
    #
    # Used by tree-based models:
    #
    #   (N, 24, 134)
    #          ↓
    #   (N, 3216)
    # --------------------------------------------------------

    X_train_flat = X_train.reshape(
        X_train.shape[0],
        -1,
    )

    X_val_flat = X_val.reshape(
        X_val.shape[0],
        -1,
    )

    X_test_flat = X_test.reshape(
        X_test.shape[0],
        -1,
    )

    expected_flat_features = (
        EXPECTED_TIN * EXPECTED_FEATURE_COUNT
    )

    if X_train_flat.shape[1] != expected_flat_features:
        raise ValueError(
            "Flattened feature count mismatch.\n"
            f"Expected: {expected_flat_features}\n"
            f"Found:    {X_train_flat.shape[1]}"
        )

    # --------------------------------------------------------
    # Target masks
    # --------------------------------------------------------

    target_valid_train = Y_mask_train.astype(bool)
    target_valid_val = Y_mask_val.astype(bool)
    target_valid_test = Y_mask_test.astype(bool)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata: Dict[str, Any] = {}

    if METADATA_PATH.exists():
        try:
            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8",
            ) as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    # --------------------------------------------------------
    # Return everything required by final experiments
    # --------------------------------------------------------

    return {
        # Sequence-format data
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test,

        # Masks
        "X_mask_train": X_mask_train,
        "Y_mask_train": Y_mask_train,
        "X_mask_val": X_mask_val,
        "Y_mask_val": Y_mask_val,
        "X_mask_test": X_mask_test,
        "Y_mask_test": Y_mask_test,

        # Boolean target validity masks
        "target_valid_train": target_valid_train,
        "target_valid_val": target_valid_val,
        "target_valid_test": target_valid_test,

        # Flattened data for RF / XGBoost
        "X_train_flat": X_train_flat,
        "X_val_flat": X_val_flat,
        "X_test_flat": X_test_flat,

        # Schema
        "feature_names": feature_names,
        "target_names": TARGET_NAMES.copy(),

        # Dimensions
        "num_features": EXPECTED_FEATURE_COUNT,
        "tin": EXPECTED_TIN,
        "tout": EXPECTED_TOUT,
        "num_targets": EXPECTED_TARGET_COUNT,

        # Scaler
        "target_scaler": target_scaler,
        "target_scaler_path": str(
            Path(
                target_scaler_path
                if target_scaler_path is not None
                else TARGET_SCALER_PATH
            )
        ),

        # Artifact paths
        "data_path": str(data_file),
        "feature_list_path": str(FEATURE_LIST_PATH),

        # Phase 7 metadata
        "metadata": metadata,
    }


# ============================================================
# TARGET INVERSE TRANSFORMATION
# ============================================================

def inverse_transform_targets(
    Y_scaled: np.ndarray,
    target_scaler,
) -> np.ndarray:
    """
    Convert scaled O3/NO2 targets back to real units.

    Supports:
        (N, 6, 2)
        (N, 2)
        (N, ...)
    """

    original_shape = Y_scaled.shape

    if Y_scaled.shape[-1] != EXPECTED_TARGET_COUNT:
        raise ValueError(
            "Target dimension mismatch.\n"
            f"Expected: {EXPECTED_TARGET_COUNT}\n"
            f"Found:    {Y_scaled.shape[-1]}"
        )

    flat = Y_scaled.reshape(
        -1,
        EXPECTED_TARGET_COUNT,
    )

    inverse = target_scaler.inverse_transform(flat)

    return inverse.reshape(original_shape)


# ============================================================
# MASKED TARGET HELPERS
# ============================================================

def masked_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return valid target values for metric computation.

    The same mask is applied to truth and prediction.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred shapes must match.\n"
            f"y_true: {y_true.shape}\n"
            f"y_pred: {y_pred.shape}"
        )

    if mask is None:
        valid = np.ones_like(
            y_true,
            dtype=bool,
        )
    else:
        if mask.shape != y_true.shape:
            raise ValueError(
                "Mask shape must match target shape.\n"
                f"mask:   {mask.shape}\n"
                f"target: {y_true.shape}"
            )

        valid = mask.astype(bool)

    return (
        y_true[valid],
        y_pred[valid],
    )


# ============================================================
# FEATURE INDEX HELPERS
# ============================================================

def get_feature_index(
    feature_names: list[str],
    feature_name: str,
) -> int:
    """Return the index of a feature in the authoritative schema."""

    if feature_name not in feature_names:
        raise KeyError(
            f"Feature not found in schema: {feature_name}"
        )

    return feature_names.index(feature_name)


def get_target_indices(
    feature_names: list[str],
) -> Dict[str, int]:
    """
    Return indices of the forecast-reference features.

    These are useful for inspecting the O3/NO2 forecast inputs.
    """

    return {
        "O3_forecast": get_feature_index(
            feature_names,
            "O3_forecast",
        ),
        "NO2_forecast": get_feature_index(
            feature_names,
            "NO2_forecast",
        ),
    }


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    splits: Optional[Dict[str, Any]] = None,
) -> None:
    """Print the final research dataset contract."""

    if splits is None:
        splits = load_splits()

    print()
    print("=" * 70)
    print("STRATOWATCH — FINAL SINGLE-SITE DATASET")
    print("=" * 70)

    print()
    print("Artifact:")
    print(f"  {splits['data_path']}")

    print()
    print("Schema:")
    print(f"  Features : {splits['num_features']}")
    print(f"  Targets  : {splits['num_targets']}")
    print(f"  Tin      : {splits['tin']}")
    print(f"  Tout     : {splits['tout']}")

    print()
    print("Targets:")
    for target in splits["target_names"]:
        print(f"  - {target}")

    print()
    print("Sequence shapes:")

    for name in [
        "X_train",
        "Y_train",
        "X_val",
        "Y_val",
        "X_test",
        "Y_test",
    ]:
        print(
            f"  {name:<8}: "
            f"{splits[name].shape}"
        )

    print()
    print("Flattened input shapes:")

    print(
        f"  X_train_flat: "
        f"{splits['X_train_flat'].shape}"
    )

    print(
        f"  X_val_flat  : "
        f"{splits['X_val_flat'].shape}"
    )

    print(
        f"  X_test_flat : "
        f"{splits['X_test_flat'].shape}"
    )

    print()
    print("Target scaler:")
    print(f"  {splits['target_scaler_path']}")

    print()
    print("Forecast feature indices:")

    indices = get_target_indices(
        splits["feature_names"]
    )

    for name, index in indices.items():
        print(f"  {name:<12}: {index}")

    print()
    print("Dataset contract: PASSED")
    print("=" * 70)
    print()


# ============================================================
# DIRECT EXECUTION TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("Loading official Phase 7 single-site artifact...")

    splits = load_splits()

    print_dataset_summary(splits)