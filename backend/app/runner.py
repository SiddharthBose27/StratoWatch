from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent

SINGLE_SITE_DIR = (
    BACKEND_DIR / "stratowatch_single_site"
)

FINAL_BASELINES_DIR = (
    SINGLE_SITE_DIR
    / "outputs"
    / "final_baselines"
)

FINAL_TRANSFORMER_DIR = (
    SINGLE_SITE_DIR
    / "outputs"
    / "final_single_site"
)

# Per-model evaluation plots are written here by
# generate_single_site_plots.py on first request.
EVALUATION_PLOTS_DIR = (
    SINGLE_SITE_DIR
    / "outputs"
    / "evaluation_plots"
)


# ============================================================
# FINAL MODEL CONTRACT
# ============================================================

SINGLE_SITE_MODELS: Dict[str, Dict[str, Any]] = {
    "xgboost": {
        "label": "XGBoost",
        "metrics_path": (
            FINAL_BASELINES_DIR
            / "xgboost"
            / "test_metrics.json"
        ),
        "prediction_path": (
            FINAL_BASELINES_DIR
            / "xgboost"
            / "test_predictions.npy"
        ),
        "truth_path": (
            FINAL_BASELINES_DIR
            / "xgboost"
            / "test_truth.npy"
        ),
        "mask_path": (
            FINAL_BASELINES_DIR
            / "xgboost"
            / "test_mask.npy"
        ),
        "model_type": "XGBRegressor",
        "target_mode": "direct_target_prediction",
        "residual_learning": False,
    },
    "random_forest": {
        "label": "Random Forest",
        "metrics_path": (
            FINAL_BASELINES_DIR
            / "random_forest"
            / "test_metrics.json"
        ),
        "prediction_path": (
            FINAL_BASELINES_DIR
            / "random_forest"
            / "test_predictions.npy"
        ),
        "truth_path": (
            FINAL_BASELINES_DIR
            / "random_forest"
            / "test_truth.npy"
        ),
        "mask_path": (
            FINAL_BASELINES_DIR
            / "random_forest"
            / "test_mask.npy"
        ),
        "model_type": "RandomForestRegressor",
        "target_mode": "direct_target_prediction",
        "residual_learning": False,
    },
    "tcn": {
        "label": "TCN",
        "metrics_path": (
            FINAL_BASELINES_DIR
            / "tcn"
            / "test_metrics.json"
        ),
        "prediction_path": (
            FINAL_BASELINES_DIR
            / "tcn"
            / "test_predictions.npy"
        ),
        "truth_path": (
            FINAL_BASELINES_DIR
            / "tcn"
            / "test_truth.npy"
        ),
        "mask_path": (
            FINAL_BASELINES_DIR
            / "tcn"
            / "test_mask.npy"
        ),
        "model_type": "TCNForecaster",
        "target_mode": "direct_target_prediction",
        "residual_learning": False,
    },
    "lstm": {
        "label": "LSTM",
        "metrics_path": (
            FINAL_BASELINES_DIR
            / "lstm"
            / "test_metrics.json"
        ),
        "prediction_path": (
            FINAL_BASELINES_DIR
            / "lstm"
            / "test_predictions.npy"
        ),
        "truth_path": (
            FINAL_BASELINES_DIR
            / "lstm"
            / "test_truth.npy"
        ),
        "mask_path": (
            FINAL_BASELINES_DIR
            / "lstm"
            / "test_mask.npy"
        ),
        "model_type": "LSTMForecaster",
        "target_mode": "direct_target_prediction",
        "residual_learning": False,
    },
    "transformer": {
        "label": "Transformer",
        "metrics_path": (
            FINAL_TRANSFORMER_DIR
            / "final_evaluation_metrics.json"
        ),
        "prediction_path": (
            FINAL_TRANSFORMER_DIR
            / "test_predictions_real_units.npy"
        ),
        "truth_path": (
            FINAL_TRANSFORMER_DIR
            / "test_truth_real_units.npy"
        ),
        "mask_path": (
            FINAL_TRANSFORMER_DIR
            / "test_mask.npy"
        ),
        "model_type": "TemporalTransformer",
        "target_mode": "direct_target_prediction",
        "residual_learning": False,
    },
}


# ============================================================
# HELPERS
# ============================================================

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in {path}"
        )

    return payload


def _first_number(
    mapping: Any,
    *keys: str,
) -> Optional[float]:
    if not isinstance(mapping, dict):
        return None

    for key in keys:
        value = mapping.get(key)

        if value is None:
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if number == number:
            return number

    return None


def _metric_block(
    block: Any,
) -> Dict[str, Optional[float]]:
    """
    Normalize:
        MAE / mae
        RMSE / rmse
        R2 / r2
        R²

    into:
        mae / rmse / r2
    """

    return {
        "mae": _first_number(
            block,
            "MAE",
            "mae",
        ),
        "rmse": _first_number(
            block,
            "RMSE",
            "rmse",
        ),
        "r2": _first_number(
            block,
            "R2",
            "r2",
            "R²",
        ),
    }


def _normalize_metrics(
    model_name: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert all five model evaluation JSON formats into
    one frontend-facing contract.
    """

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_source = payload.get("overall")

    # XGBoost / RF / TCN / LSTM:
    # overall is normally directly available.
    #
    # Some artifacts may wrap metrics under "test".
    if not isinstance(overall_source, dict):
        test_block = payload.get("test")

        if isinstance(test_block, dict):
            overall_source = test_block.get(
                "overall",
                test_block,
            )

    overall = _metric_block(
        overall_source
    )

    # --------------------------------------------------------
    # Per-target
    # --------------------------------------------------------

    target_source = (
        payload.get("per_target")
    )

    if not isinstance(target_source, dict):
        target_source = payload.get(
            "targets",
            {},
        )

    per_target: Dict[str, Any] = {}

    if isinstance(target_source, dict):
        for target_name, block in target_source.items():
            per_target[str(target_name)] = _metric_block(
                block
            )

    # --------------------------------------------------------
    # Horizon-wise
    # --------------------------------------------------------

    horizon_source = (
        payload.get("horizon_wise")
    )

    if not isinstance(horizon_source, dict):
        horizon_source = payload.get(
            "horizons",
            {},
        )

    horizon_wise: Dict[str, Any] = {}

    if isinstance(horizon_source, dict):
        for horizon, block in horizon_source.items():
            horizon_wise[str(horizon)] = _metric_block(
                block
            )

    return {
        "model": model_name,
        "overall": overall,
        "per_target": per_target,
        "horizon_wise": horizon_wise,
        "evaluation_type": payload.get(
            "evaluation_type",
            "final_test_real_units",
        ),
        "dataset": (
            payload.get("dataset")
            or payload.get("metadata", {}).get("dataset")
        ),
        "model_type": (
            payload.get("model_type")
            or payload.get("metadata", {}).get("model_type")
            or SINGLE_SITE_MODELS[
                model_name
            ]["model_type"]
        ),
        "target_mode": (
            payload.get("target_mode")
            or payload.get("metadata", {}).get("target_mode")
            or SINGLE_SITE_MODELS[
                model_name
            ]["target_mode"]
        ),
        "residual_learning": (
            payload.get("residual_learning")
            if "residual_learning" in payload
            else payload.get(
                "metadata",
                {},
            ).get(
                "residual_learning",
                SINGLE_SITE_MODELS[
                    model_name
                ]["residual_learning"],
            )
        ),
    }


def _encode_file(
    path: Path,
) -> str:
    with path.open("rb") as file:
        return base64.b64encode(
            file.read()
        ).decode("utf-8")


# ============================================================
# MODEL-SPECIFIC PLOT GENERATION
# ============================================================

def _get_plot_module():
    """
    Import generate_model_plots from generate_single_site_plots.py.

    This lives in the same app/ directory as runner.py.
    """
    # Ensure the app/ directory is on sys.path so we can import
    # the sibling module without triggering circular imports.
    app_dir_str = str(APP_DIR)
    if app_dir_str not in sys.path:
        sys.path.insert(0, app_dir_str)

    try:
        # Import by file path to avoid conflicts with package imports.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_single_site_plots",
            APP_DIR / "generate_single_site_plots.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.generate_model_plots
    except Exception as exc:
        raise RuntimeError(
            f"Could not import generate_single_site_plots: {exc}"
        ) from exc


def _collect_model_specific_plots(
    model_name: str,
) -> List[Dict[str, str]]:
    """
    Return base64-encoded evaluation plots for the selected model only.

    Strategy:
      1. Check if model-specific plots already exist in
         evaluation_plots/{model_name}/.
      2. If not, generate them on-demand from the frozen .npy arrays
         via generate_single_site_plots.generate_model_plots().
      3. Encode and return only the plots for this model.

    Plot names returned are keys like:
        o3_actual_vs_predicted
        no2_actual_vs_predicted
        o3_scatter
        no2_scatter
        etc.
    which the frontend already handles correctly.
    """

    model_plot_dir = EVALUATION_PLOTS_DIR / model_name

    # Check if plots already exist (cache hit).
    existing_pngs: List[Path] = []
    if model_plot_dir.exists():
        existing_pngs = sorted(model_plot_dir.glob("*.png"))

    if not existing_pngs:
        # Generate model-specific plots from frozen .npy artifacts.
        try:
            generate_model_plots = _get_plot_module()
            generated = generate_model_plots(model_name)
            # generated is a dict: {plot_key: absolute_path_str}
            # Collect paths from what was just generated.
            existing_pngs = [
                Path(p)
                for p in generated.values()
                if Path(p).exists()
            ]
        except Exception:
            # Plot generation failed (e.g., missing .npy artifacts).
            # Return empty list; the frontend handles the empty state.
            return []

    return [
        {
            # Strip the model prefix from the filename to get the
            # plot key the frontend expects, e.g.:
            #   "xgboost_o3_actual_vs_predicted.png"
            #   → "o3_actual_vs_predicted.png"
            "name": _strip_model_prefix(path.name, model_name),
            "b64": _encode_file(path),
        }
        for path in sorted(existing_pngs)
        if path.exists()
    ]


def _strip_model_prefix(filename: str, model_name: str) -> str:
    """
    Remove the model-name prefix from plot filenames so the frontend
    can match them against its PLOT_LABELS dictionary.

    Examples:
        xgboost_o3_actual_vs_predicted.png → o3_actual_vs_predicted.png
        random_forest_no2_scatter.png      → no2_scatter.png
        lstm_training_loss.png             → lstm_training_loss.png
          (no match → kept as-is)
    """
    safe_prefix = (
        model_name
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_"
    )

    if filename.startswith(safe_prefix):
        return filename[len(safe_prefix):]

    return filename


def _artifact_status(
    config: Dict[str, Any],
) -> Dict[str, bool]:
    return {
        "metrics": Path(
            config["metrics_path"]
        ).exists(),
        "predictions": Path(
            config["prediction_path"]
        ).exists(),
        "truth": Path(
            config["truth_path"]
        ).exists(),
        "mask": Path(
            config["mask_path"]
        ).exists(),
    }


# ============================================================
# PUBLIC API FUNCTION
# ============================================================

def run_single_site_real(
    uploaded_csv_path: Optional[str],
    model_name: str,
    include_confusion: bool = False,
) -> Dict[str, Any]:
    """
    Return the final frozen single-site evaluation artifact for
    the SELECTED MODEL only.

    Plots are generated on-demand from the model's frozen .npy
    prediction/truth/mask arrays and are specific to that model.
    Subsequent calls for the same model reuse cached plot files.

    Important:
        The uploaded CSV is accepted by the API for interface
        compatibility, but it is NOT substituted into the official
        Phase 7 research evaluation.

    No training occurs here.
    No model checkpoint is modified.
    """

    del include_confusion

    if model_name not in SINGLE_SITE_MODELS:
        return {
            "error": (
                f"Unknown model_name: {model_name}. "
                f"Expected one of: "
                f"{', '.join(SINGLE_SITE_MODELS.keys())}"
            )
        }

    config = SINGLE_SITE_MODELS[
        model_name
    ]

    metrics_path = Path(
        config["metrics_path"]
    )

    try:
        payload = _load_json(
            metrics_path
        )

        metrics = _normalize_metrics(
            model_name,
            payload,
        )

    except Exception as exc:
        return {
            "error": (
                f"Unable to load final evaluation artifact "
                f"for {model_name}: {exc}"
            ),
            "model": model_name,
            "artifact_status": _artifact_status(
                config
            ),
        }

    artifact_status = _artifact_status(
        config
    )

    warnings: List[str] = []

    missing = [
        name
        for name, exists in artifact_status.items()
        if not exists
    ]

    if missing:
        warnings.append(
            "Some saved evaluation artifacts are missing: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Collect model-specific plots ONLY
    # --------------------------------------------------------
    plots = _collect_model_specific_plots(
        model_name
    )

    return {
        "success": True,
        "model": model_name,
        "metrics": metrics,
        "plots": plots,
        "warnings": warnings,
        "evaluation_source": (
            "official_phase_7_frozen_test_artifact"
        ),
        "uploaded_file_used_for_research_evaluation": False,
        "evaluation_only": True,
        "retraining": False,
        "artifact_status": artifact_status,
    }


# ============================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================

def run_single_site(
    model_name: str,
    uploaded_csv_path: Optional[str] = None,
    include_confusion: bool = False,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for older callers.
    """

    return run_single_site_real(
        uploaded_csv_path=uploaded_csv_path,
        model_name=model_name,
        include_confusion=include_confusion,
    )