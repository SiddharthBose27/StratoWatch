from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

PROJECT_DIR = os.path.join(
    ROOT,
    "stratowatch_multi_site",
)

FINAL_EVAL_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "final_evaluation",
)

MULTISITE_PLOT_DIR = os.path.join(
    FINAL_EVAL_DIR,
    "plots",
)

FINAL_RESULTS_JSON = os.path.join(
    FINAL_EVAL_DIR,
    "final_multisite_results.json",
)

TARGET_NAMES = ["O3", "NO2"]
TOUT = 6
EXPECTED_SITE_COUNT = 7


MODEL_TO_CMD = {
    "st_transformer": [
        "-m",
        "src.eval_baseline_realunits",
    ],
    "graph_st_static": [
        "-m",
        "src.eval_graph_realunits",
        "--model",
        "static",
    ],
    "graph_st_dynamic_wind": [
        "-m",
        "src.eval_graph_realunits",
        "--model",
        "dynamic",
    ],
}


_MODEL_ARTIFACTS = {
    "st_transformer": {
        "pred": "st_test_predictions_real.npy",
        "truth": "st_test_truth_real.npy",
        "mask": "st_test_mask.npy",
    },
    "graph_st_static": {
        "pred": "static_test_predictions_real.npy",
        "truth": "static_test_truth_real.npy",
        "mask": "static_test_mask.npy",
    },
    "graph_st_dynamic_wind": {
        "pred": "dynamic_test_predictions_real.npy",
        "truth": "dynamic_test_truth_real.npy",
        "mask": "dynamic_test_mask.npy",
    },
}


def _safe_model_name(model_name: str) -> str:
    if model_name not in MODEL_TO_CMD:
        raise ValueError(
            f"Unsupported multi-site model: {model_name}"
        )
    return model_name


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in {path}"
        )

    return data


def _find_artifact(
    directory: str,
    filename: str,
) -> Optional[str]:
    direct = os.path.join(directory, filename)

    if os.path.exists(direct):
        return direct

    matches = glob(
        os.path.join(
            directory,
            "**",
            filename,
        ),
        recursive=True,
    )

    return matches[0] if matches else None


def _load_model_artifacts(
    model_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    model_name = _safe_model_name(model_name)

    spec = _MODEL_ARTIFACTS[model_name]

    pred_path = _find_artifact(
        FINAL_EVAL_DIR,
        spec["pred"],
    )
    truth_path = _find_artifact(
        FINAL_EVAL_DIR,
        spec["truth"],
    )
    mask_path = _find_artifact(
        FINAL_EVAL_DIR,
        spec["mask"],
    )

    if not pred_path or not truth_path:
        return None

    predictions = np.load(
        pred_path,
        allow_pickle=False,
    )

    actuals = np.load(
        truth_path,
        allow_pickle=False,
    )

    mask = None

    if mask_path:
        mask = np.load(
            mask_path,
            allow_pickle=False,
        )

    return predictions, actuals, mask


def _encode_plot(fig) -> str:
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def _normalise_plot_item(
    title: str,
    image_base64: str,
) -> Dict[str, Any]:
    return {
        "title": title,
        "image": image_base64,
    }


def _plot_series(
    title: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    ylabel: str,
) -> str:
    fig = plt.figure(
        figsize=(10, 4.5)
    )

    ax = fig.add_subplot(111)

    x = np.arange(
        min(
            len(actual),
            len(predicted),
        )
    )

    ax.plot(
        x,
        np.asarray(actual)[: len(x)],
        label="Actual",
        linewidth=2,
    )

    ax.plot(
        x,
        np.asarray(predicted)[: len(x)],
        label="Predicted",
        linewidth=2,
    )

    ax.set_title(title)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)

    return _encode_plot(fig)


def _generate_multisite_plots(
    model_name: str,
) -> List[Dict[str, Any]]:
    artifacts = _load_model_artifacts(
        model_name
    )

    if artifacts is None:
        return []

    predictions, actuals, mask = artifacts

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if predictions.ndim != 4:
        return []

    if actuals.ndim != 4:
        return []

    model_plot_dir = os.path.join(
        MULTISITE_PLOT_DIR,
        model_name,
    )

    os.makedirs(
        model_plot_dir,
        exist_ok=True,
    )

    plots: List[Dict[str, Any]] = []

    # Expected shape:
    # (samples, horizons, sites, targets)
    #
    # The frozen evaluation artifacts are already
    # research-evaluation outputs, so these plots
    # are generated directly from those artifacts.

    sample_count = predictions.shape[0]

    if sample_count == 0:
        return []

    sample_index = 0

    for target_index, target_name in enumerate(
        TARGET_NAMES
    ):
        if target_index >= predictions.shape[-1]:
            continue

        actual = actuals[
            sample_index,
            :,
            :,
            target_index,
        ]

        predicted = predictions[
            sample_index,
            :,
            :,
            target_index,
        ]

        if actual.ndim == 2:
            actual_series = np.nanmean(
                actual,
                axis=1,
            )

            predicted_series = np.nanmean(
                predicted,
                axis=1,
            )
        else:
            actual_series = np.asarray(actual).reshape(-1)
            predicted_series = np.asarray(predicted).reshape(-1)

        image_base64 = _plot_series(
            title=(
                f"{model_name} — {target_name} "
                "Frozen Evaluation"
            ),
            actual=actual_series,
            predicted=predicted_series,
            ylabel=f"{target_name} (µg/m³)",
        )

        filename = (
            f"{model_name}_{target_name.lower()}.png"
        )

        output_path = os.path.join(
            model_plot_dir,
            filename,
        )

        with open(
            output_path,
            "wb",
        ) as f:
            f.write(
                base64.b64decode(
                    image_base64
                )
            )

        plots.append(
            _normalise_plot_item(
                title=(
                    f"{model_name} — "
                    f"{target_name}"
                ),
                image_base64=image_base64,
            )
        )

    return plots


def _plot_live_fallback(
    sites: List[str],
    predictions: np.ndarray,
    actuals: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if predictions.ndim != 3:
        return []

    plots: List[Dict[str, Any]] = []

    horizon_count = predictions.shape[0]

    for target_index, target_name in enumerate(
        TARGET_NAMES
    ):
        if target_index >= predictions.shape[-1]:
            continue

        fig = plt.figure(
            figsize=(10, 4.5)
        )

        ax = fig.add_subplot(111)

        x = np.arange(
            1,
            horizon_count + 1,
        )

        for site_index in range(
            predictions.shape[1]
        ):
            site_name = (
                sites[site_index]
                if site_index < len(sites)
                else f"Site {site_index + 1}"
            )

            ax.plot(
                x,
                predictions[
                    :,
                    site_index,
                    target_index,
                ],
                marker="o",
                linewidth=2,
                label=site_name,
            )

        if actuals is not None:
            actual_array = np.asarray(
                actuals,
                dtype=float,
            )

            if (
                actual_array.ndim == 3
                and actual_array.shape[0]
                >= horizon_count
            ):
                for site_index in range(
                    min(
                        actual_array.shape[1],
                        predictions.shape[1],
                    )
                ):
                    site_name = (
                        sites[site_index]
                        if site_index < len(sites)
                        else f"Site {site_index + 1}"
                    )

                    ax.plot(
                        x,
                        actual_array[
                            :horizon_count,
                            site_index,
                            target_index,
                        ],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label=f"{site_name} Actual",
                    )

        ax.set_title(
            f"Request-Local Forecast — {target_name}"
        )
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel(
            f"{target_name} (µg/m³)"
        )
        ax.grid(alpha=0.25)

        if len(sites) <= 7:
            ax.legend(
                fontsize=8,
                ncol=2,
            )

        plots.append(
            _normalise_plot_item(
                title=(
                    f"Request-Local "
                    f"{target_name} Forecast"
                ),
                image_base64=_encode_plot(fig),
            )
        )

    return plots


def _clean_site_name(
    path: str,
    index: int,
) -> str:
    filename = os.path.basename(path)

    stem = os.path.splitext(
        filename
    )[0].strip()

    stem = re.sub(
        r"[_\-]+",
        " ",
        stem,
    ).strip()

    return stem or f"Site {index + 1}"


def _read_numeric_column(
    df: pd.DataFrame,
    column: str,
) -> Optional[np.ndarray]:
    if column not in df.columns:
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return None

    return values


def _latest_values(
    values: np.ndarray,
    count: int,
) -> Optional[np.ndarray]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < count:
        return None

    return values[-count:]


def _request_local_fallback(
    uploaded_csv_paths: List[str],
    uploaded_site_names: List[str],
) -> Dict[str, Any]:
    predictions_by_site: List[np.ndarray] = []
    actuals_by_site: List[Optional[np.ndarray]] = []
    resolved_sites: List[str] = []

    for index, csv_path in enumerate(
        uploaded_csv_paths
    ):
        df = pd.read_csv(csv_path)

        site_name = (
            uploaded_site_names[index]
            if index < len(uploaded_site_names)
            and uploaded_site_names[index]
            else _clean_site_name(
                csv_path,
                index,
            )
        )

        resolved_sites.append(
            site_name
        )

        site_prediction = np.zeros(
            (
                TOUT,
                len(TARGET_NAMES),
            ),
            dtype=float,
        )

        site_actual: Optional[np.ndarray] = None

        for target_index, target_name in enumerate(
            TARGET_NAMES
        ):
            forecast_column = (
                f"{target_name}_forecast"
            )

            target_column = target_name

            forecast_values = (
                _read_numeric_column(
                    df,
                    forecast_column,
                )
            )

            target_values = (
                _read_numeric_column(
                    df,
                    target_column,
                )
            )

            selected_prediction = None

            if forecast_values is not None:
                selected_prediction = _latest_values(
                    forecast_values,
                    TOUT,
                )

            if selected_prediction is None:
                selected_prediction = (
                    _latest_values(
                        target_values,
                        TOUT,
                    )
                    if target_values is not None
                    else None
                )

            if selected_prediction is None:
                raise ValueError(
                    f"Could not obtain {target_name} "
                    f"forecast/persistence values "
                    f"from {csv_path}"
                )

            site_prediction[
                :,
                target_index,
            ] = selected_prediction

        # If actual target columns are present,
        # retain them for current-request metrics.
        actual_columns_available = all(
            target_name in df.columns
            for target_name in TARGET_NAMES
        )

        if actual_columns_available:
            actual_target_arrays = []

            for target_name in TARGET_NAMES:
                target_values = (
                    _read_numeric_column(
                        df,
                        target_name,
                    )
                )

                selected_actual = (
                    _latest_values(
                        target_values,
                        TOUT,
                    )
                    if target_values is not None
                    else None
                )

                if selected_actual is None:
                    actual_target_arrays = []
                    break

                actual_target_arrays.append(
                    selected_actual
                )

            if actual_target_arrays:
                site_actual = np.stack(
                    actual_target_arrays,
                    axis=-1,
                )

        predictions_by_site.append(
            site_prediction
        )

        actuals_by_site.append(
            site_actual
        )

    if not predictions_by_site:
        raise ValueError(
            "No uploaded site datasets were provided."
        )

    predictions = np.stack(
        predictions_by_site,
        axis=1,
    )

    metrics: Dict[str, Any] = {}

    if all(
        actual is not None
        for actual in actuals_by_site
    ):
        actuals = np.stack(
            [
                actual
                for actual in actuals_by_site
                if actual is not None
            ],
            axis=1,
        )

        error = predictions - actuals

        for target_index, target_name in enumerate(
            TARGET_NAMES
        ):
            target_error = error[
                :,
                :,
                target_index,
            ]

            target_actual = actuals[
                :,
                :,
                target_index,
            ]

            mae = float(
                np.mean(
                    np.abs(
                        target_error
                    )
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        target_error ** 2
                    )
                )
            )

            actual_mean = float(
                np.mean(target_actual)
            )

            ss_res = float(
                np.sum(
                    target_error ** 2
                )
            )

            ss_tot = float(
                np.sum(
                    (
                        target_actual
                        - actual_mean
                    )
                    ** 2
                )
            )

            r2 = (
                1.0 - ss_res / ss_tot
                if ss_tot > 0
                else None
            )

            metrics[target_name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }

        plot_actuals = actuals

    else:
        actuals = None
        plot_actuals = None

    plots = _plot_live_fallback(
        sites=resolved_sites,
        predictions=predictions,
        actuals=plot_actuals,
    )

    return {
        "sites": resolved_sites,
        "targets": TARGET_NAMES,
        "horizons": list(
            range(
                1,
                TOUT + 1,
            )
        ),
        "predictions": predictions.tolist(),
        "actuals": (
            actuals.tolist()
            if actuals is not None
            else None
        ),
        "metrics": metrics,
        "plots": plots,
    }


def _get_frozen_metrics(
    model_name: str,
) -> Dict[str, Any]:
    if not os.path.exists(
        FINAL_RESULTS_JSON
    ):
        raise FileNotFoundError(
            "Frozen multi-site results file "
            f"was not found: {FINAL_RESULTS_JSON}"
        )

    results = _load_json(
        FINAL_RESULTS_JSON
    )

    # Support either a model-keyed structure
    # or a top-level "models" structure.
    models = results.get(
        "models",
        results,
    )

    if not isinstance(
        models,
        dict,
    ):
        raise ValueError(
            "Invalid frozen multi-site results format."
        )

    if model_name not in models:
        raise KeyError(
            f"Frozen metrics for {model_name} "
            "were not found."
        )

    model_results = models[
        model_name
    ]

    if isinstance(
        model_results,
        dict
    ):
        return model_results

    raise ValueError(
        f"Invalid frozen metrics for {model_name}."
    )


def run_multisite(
    model_name: str,
    site_count: int,
    uploaded_csv_paths: Optional[List[str]] = None,
    uploaded_site_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    model_name = _safe_model_name(
        model_name
    )

    uploaded_csv_paths = (
        uploaded_csv_paths or []
    )

    uploaded_site_names = (
        uploaded_site_names or []
    )

    # ---------------------------------------------------------
    # Official frozen research configuration
    # ---------------------------------------------------------
    #
    # The frozen multi-site research checkpoints were trained
    # and evaluated for exactly 7 sites.
    #
    # Uploaded files are intentionally NOT substituted into
    # the official research evaluation path.
    #
    # This preserves the published/frozen research results.
    # ---------------------------------------------------------
    if site_count == EXPECTED_SITE_COUNT:
        metrics = _get_frozen_metrics(
            model_name
        )

        plots = _generate_multisite_plots(
            model_name
        )

        return {
            "ok": True,
            "mode": "checkpoint",
            "inference_mode": "trained_checkpoint",
            "model": model_name,
            "site_count": EXPECTED_SITE_COUNT,
            "sites": uploaded_site_names or [],
            "targets": TARGET_NAMES,
            "horizons": list(
                range(
                    1,
                    TOUT + 1,
                )
            ),
            "metrics": metrics,
            "plots": plots,
            "warnings": [],
            "uploaded_files_used_for_research_evaluation": False,
        }

    # ---------------------------------------------------------
    # Request-local mode for other site counts
    # ---------------------------------------------------------
    #
    # This is a valid product/demo path, but it is NOT the
    # frozen 7-site research checkpoint evaluation.
    # ---------------------------------------------------------
    if site_count < 1:
        raise ValueError(
            "site_count must be at least 1."
        )

    if len(uploaded_csv_paths) != site_count:
        raise ValueError(
            f"Expected {site_count} uploaded CSV "
            f"files, but received "
            f"{len(uploaded_csv_paths)}."
        )

    fallback = _request_local_fallback(
        uploaded_csv_paths=uploaded_csv_paths,
        uploaded_site_names=uploaded_site_names,
    )

    return {
        "ok": True,
        "mode": "request_local_fallback",
        "inference_mode": "persistence_fallback",
        "model": model_name,
        "site_count": site_count,
        "warning": (
            "Request-local forecast mode: the frozen "
            "research checkpoints support the 7-site "
            "configuration. Because you selected "
            f"{site_count} sites, this request used a "
            "local forecast/persistence fallback based "
            "on the uploaded datasets."
        ),
        **fallback,
        "uploaded_files_used_for_research_evaluation": True,
    }