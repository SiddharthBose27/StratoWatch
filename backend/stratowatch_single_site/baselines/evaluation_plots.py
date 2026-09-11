"""
StratoWatch — Frozen Single-Site Evaluation Plotting

Shared plotting utilities for final evaluation scripts.

IMPORTANT:
    These plots are generated ONLY from frozen-model predictions
    and the official Phase 7 test truth.

    No training is performed.
    No predictions are modified.
    No metrics are modified.

Expected arrays:
    truth       : (N, 6, 2)
    predictions : (N, 6, 2)
    mask        : (N, 6, 2)

Targets:
    O3
    NO2

Generated figures:
    1. O3 actual vs predicted
    2. NO2 actual vs predicted
    3. O3 residual distribution
    4. NO2 residual distribution
    5. O3 scatter actual vs predicted
    6. NO2 scatter actual vs predicted
    7. Horizon-wise MAE/RMSE
    8. O3 error by forecast horizon
    9. NO2 error by forecast horizon
    10. Target-wise MAE/RMSE
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


TARGET_NAMES = [
    "O3",
    "NO2",
]

TARGET_DISPLAY_NAMES = {
    "O3": "O₃",
    "NO2": "NO₂",
}

TOUT = 6


# ============================================================
# VALIDATION
# ============================================================

def _validate_arrays(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> None:

    if truth.shape != predictions.shape:
        raise ValueError(
            "Truth/prediction shape mismatch: "
            f"{truth.shape} vs {predictions.shape}"
        )

    if truth.shape != mask.shape:
        raise ValueError(
            "Truth/mask shape mismatch: "
            f"{truth.shape} vs {mask.shape}"
        )

    if truth.ndim != 3:
        raise ValueError(
            "Expected arrays with shape (N, horizon, target), "
            f"got {truth.shape}"
        )

    if truth.shape[1] != TOUT:
        raise ValueError(
            f"Expected {TOUT} forecast horizons, "
            f"got {truth.shape[1]}"
        )

    if truth.shape[2] != len(TARGET_NAMES):
        raise ValueError(
            f"Expected {len(TARGET_NAMES)} targets, "
            f"got {truth.shape[2]}"
        )

    if not np.isfinite(truth).all():
        raise ValueError(
            "Truth contains non-finite values."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Predictions contain non-finite values."
        )


# ============================================================
# HELPERS
# ============================================================

def _safe_valid_values(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
):
    valid = mask.astype(bool)

    return (
        truth[valid],
        predictions[valid],
    )


def _save(
    fig,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def _target_label(target_name: str) -> str:
    return TARGET_DISPLAY_NAMES.get(
        target_name,
        target_name,
    )


# ============================================================
# 1. ACTUAL VS PREDICTED LINE
# ============================================================

def plot_actual_vs_predicted(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    target_name: str,
    output_path: Path,
    max_points: int = 500,
) -> None:
    """
    Plot a chronological sample of actual and predicted values.

    The plotted sequence is flattened in forecast order:
        sample 1 H+1
        sample 1 H+2
        ...
        sample N H+6

    Only valid masked observations are shown.
    """

    target_truth = truth[:, :, target_idx]
    target_pred = predictions[:, :, target_idx]
    target_mask = mask[:, :, target_idx].astype(bool)

    true_values = target_truth[target_mask]
    pred_values = target_pred[target_mask]

    if len(true_values) == 0:
        return

    n = min(
        max_points,
        len(true_values),
    )

    x = np.arange(n)

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        x,
        true_values[:n],
        label="Actual",
        linewidth=1.4,
    )

    ax.plot(
        x,
        pred_values[:n],
        label="Predicted",
        linewidth=1.2,
    )

    ax.set_title(
        f"{_target_label(target_name)} — Actual vs Predicted"
    )

    ax.set_xlabel(
        "Forecasted test observations"
    )

    ax.set_ylabel(
        "Concentration"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    _save(
        fig,
        output_path,
    )


# ============================================================
# 2. ACTUAL VS PREDICTED SCATTER
# ============================================================

def plot_scatter(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    target_name: str,
    output_path: Path,
    max_points: int = 10000,
) -> None:

    target_truth = truth[:, :, target_idx]
    target_pred = predictions[:, :, target_idx]
    target_mask = mask[:, :, target_idx].astype(bool)

    true_values = target_truth[target_mask]
    pred_values = target_pred[target_mask]

    if len(true_values) == 0:
        return

    if len(true_values) > max_points:

        indices = np.linspace(
            0,
            len(true_values) - 1,
            max_points,
            dtype=int,
        )

        true_values = true_values[
            indices
        ]

        pred_values = pred_values[
            indices
        ]

    minimum = min(
        np.min(true_values),
        np.min(pred_values),
    )

    maximum = max(
        np.max(true_values),
        np.max(pred_values),
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        true_values,
        pred_values,
        s=10,
        alpha=0.25,
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction",
    )

    ax.set_title(
        f"{_target_label(target_name)} — Actual vs Predicted"
    )

    ax.set_xlabel(
        "Actual"
    )

    ax.set_ylabel(
        "Predicted"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    _save(
        fig,
        output_path,
    )


# ============================================================
# 3. RESIDUAL DISTRIBUTION
# ============================================================

def plot_residual_distribution(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    target_name: str,
    output_path: Path,
) -> None:

    target_truth = truth[:, :, target_idx]
    target_pred = predictions[:, :, target_idx]
    target_mask = mask[:, :, target_idx].astype(bool)

    residuals = (
        target_pred[target_mask]
        - target_truth[target_mask]
    )

    if len(residuals) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        residuals,
        bins=60,
        alpha=0.8,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_title(
        f"{_target_label(target_name)} — Residual Distribution"
    )

    ax.set_xlabel(
        "Prediction − Actual"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        alpha=0.25
    )

    _save(
        fig,
        output_path,
    )


# ============================================================
# 4. HORIZON-WISE MAE / RMSE
# ============================================================

def plot_horizon_performance(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:

    horizons = np.arange(
        1,
        TOUT + 1,
    )

    mae_values = []
    rmse_values = []

    for h in range(TOUT):

        valid = mask[
            :,
            h,
            :,
        ].astype(bool)

        true_values = truth[
            :,
            h,
            :,
        ][valid]

        pred_values = predictions[
            :,
            h,
            :,
        ][valid]

        if len(true_values) == 0:
            mae_values.append(
                np.nan
            )

            rmse_values.append(
                np.nan
            )

            continue

        errors = (
            pred_values
            - true_values
        )

        mae_values.append(
            float(
                np.mean(
                    np.abs(errors)
                )
            )
        )

        rmse_values.append(
            float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            )
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        horizons,
        mae_values,
        marker="o",
        linewidth=1.8,
        label="MAE",
    )

    ax.plot(
        horizons,
        rmse_values,
        marker="o",
        linewidth=1.8,
        label="RMSE",
    )

    ax.set_title(
        "Forecast Horizon Performance"
    )

    ax.set_xlabel(
        "Forecast horizon (hours)"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_xticks(
        horizons
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    _save(
        fig,
        output_path,
    )


# ============================================================
# 5. TARGET ERROR BY HORIZON
# ============================================================

def plot_target_error_by_horizon(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    target_name: str,
    output_path: Path,
) -> None:

    horizons = np.arange(
        1,
        TOUT + 1,
    )

    mae_values = []
    rmse_values = []

    for h in range(TOUT):

        true_values = truth[
            :,
            h,
            target_idx,
        ]

        pred_values = predictions[
            :,
            h,
            target_idx,
        ]

        valid = mask[
            :,
            h,
            target_idx,
        ].astype(bool)

        true_values = true_values[
            valid
        ]

        pred_values = pred_values[
            valid
        ]

        if len(true_values) == 0:

            mae_values.append(
                np.nan
            )

            rmse_values.append(
                np.nan
            )

            continue

        errors = (
            pred_values
            - true_values
        )

        mae_values.append(
            float(
                np.mean(
                    np.abs(errors)
                )
            )
        )

        rmse_values.append(
            float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            )
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        horizons,
        mae_values,
        marker="o",
        linewidth=1.8,
        label="MAE",
    )

    ax.plot(
        horizons,
        rmse_values,
        marker="o",
        linewidth=1.8,
        label="RMSE",
    )

    ax.set_title(
        f"{_target_label(target_name)} — Error by Forecast Horizon"
    )

    ax.set_xlabel(
        "Forecast horizon (hours)"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_xticks(
        horizons
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    _save(
        fig,
        output_path,
    )


# ============================================================
# 6. TARGET-WISE PERFORMANCE
# ============================================================

def plot_target_performance(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:

    mae_values = []
    rmse_values = []

    for target_idx in range(
        len(TARGET_NAMES)
    ):

        target_truth = truth[
            :,
            :,
            target_idx,
        ]

        target_pred = predictions[
            :,
            :,
            target_idx,
        ]

        target_mask = mask[
            :,
            :,
            target_idx,
        ].astype(bool)

        true_values = target_truth[
            target_mask
        ]

        pred_values = target_pred[
            target_mask
        ]

        if len(true_values) == 0:

            mae_values.append(
                np.nan
            )

            rmse_values.append(
                np.nan
            )

            continue

        errors = (
            pred_values
            - true_values
        )

        mae_values.append(
            float(
                np.mean(
                    np.abs(errors)
                )
            )
        )

        rmse_values.append(
            float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            )
        )

    x = np.arange(
        len(TARGET_NAMES)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        mae_values,
        width,
        label="MAE",
    )

    ax.bar(
        x + width / 2,
        rmse_values,
        width,
        label="RMSE",
    )

    ax.set_title(
        "Target-wise Error"
    )

    ax.set_xlabel(
        "Target"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            _target_label(name)
            for name in TARGET_NAMES
        ]
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    _save(
        fig,
        output_path,
    )


# ============================================================
# MASTER FUNCTION
# ============================================================

def generate_evaluation_plots(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> Dict[str, str]:
    """
    Generate the complete frozen-model evaluation plot set.

    Returns:
        Dictionary mapping plot name -> absolute path.
    """

    _validate_arrays(
        truth,
        predictions,
        mask,
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = {}

    safe_model_name = (
        model_name
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    # --------------------------------------------------------
    # Target-specific plots
    # --------------------------------------------------------

    for target_idx, target_name in enumerate(
        TARGET_NAMES
    ):

        actual_pred_path = (
            output_dir
            / f"{safe_model_name}_{target_name.lower()}_actual_vs_predicted.png"
        )

        plot_actual_vs_predicted(
            truth=truth,
            predictions=predictions,
            mask=mask,
            target_idx=target_idx,
            target_name=target_name,
            output_path=actual_pred_path,
        )

        if actual_pred_path.exists():
            generated[
                f"{target_name}_actual_vs_predicted"
            ] = str(
                actual_pred_path
            )

        # ----------------------------------------------------
        # Scatter
        # ----------------------------------------------------

        scatter_path = (
            output_dir
            / f"{safe_model_name}_{target_name.lower()}_scatter.png"
        )

        plot_scatter(
            truth=truth,
            predictions=predictions,
            mask=mask,
            target_idx=target_idx,
            target_name=target_name,
            output_path=scatter_path,
        )

        if scatter_path.exists():
            generated[
                f"{target_name}_scatter"
            ] = str(
                scatter_path
            )

        # ----------------------------------------------------
        # Residuals
        # ----------------------------------------------------

        residual_path = (
            output_dir
            / f"{safe_model_name}_{target_name.lower()}_residual_distribution.png"
        )

        plot_residual_distribution(
            truth=truth,
            predictions=predictions,
            mask=mask,
            target_idx=target_idx,
            target_name=target_name,
            output_path=residual_path,
        )

        if residual_path.exists():
            generated[
                f"{target_name}_residual_distribution"
            ] = str(
                residual_path
            )

        # ----------------------------------------------------
        # Error by horizon
        # ----------------------------------------------------

        horizon_error_path = (
            output_dir
            / f"{safe_model_name}_{target_name.lower()}_error_by_horizon.png"
        )

        plot_target_error_by_horizon(
            truth=truth,
            predictions=predictions,
            mask=mask,
            target_idx=target_idx,
            target_name=target_name,
            output_path=horizon_error_path,
        )

        if horizon_error_path.exists():
            generated[
                f"{target_name}_error_by_horizon"
            ] = str(
                horizon_error_path
            )

    # --------------------------------------------------------
    # Global horizon performance
    # --------------------------------------------------------

    horizon_path = (
        output_dir
        / f"{safe_model_name}_horizon_performance.png"
    )

    plot_horizon_performance(
        truth=truth,
        predictions=predictions,
        mask=mask,
        output_path=horizon_path,
    )

    if horizon_path.exists():
        generated[
            "horizon_performance"
        ] = str(
            horizon_path
        )

    # --------------------------------------------------------
    # Target-wise performance
    # --------------------------------------------------------

    target_path = (
        output_dir
        / f"{safe_model_name}_target_performance.png"
    )

    plot_target_performance(
        truth=truth,
        predictions=predictions,
        mask=mask,
        output_path=target_path,
    )

    if target_path.exists():
        generated[
            "target_performance"
        ] = str(
            target_path
        )

    print()
    print(
        f"Generated {len(generated)} evaluation plots "
        f"for {model_name}."
    )

    for name, path in generated.items():
        print(
            f"  ✓ {name}: {path}"
        )

    return generated