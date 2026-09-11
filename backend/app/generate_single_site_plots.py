from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

SINGLE_SITE_DIR = BACKEND_DIR / "stratowatch_single_site"

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

PLOTS_ROOT = (
    SINGLE_SITE_DIR
    / "outputs"
    / "evaluation_plots"
)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict[str, Path]] = {
    "xgboost": {
        "directory": FINAL_BASELINES_DIR / "xgboost",
        "predictions": FINAL_BASELINES_DIR
        / "xgboost"
        / "test_predictions.npy",
        "truth": FINAL_BASELINES_DIR
        / "xgboost"
        / "test_truth.npy",
        "mask": FINAL_BASELINES_DIR
        / "xgboost"
        / "test_mask.npy",
    },
    "random_forest": {
        "directory": FINAL_BASELINES_DIR / "random_forest",
        "predictions": FINAL_BASELINES_DIR
        / "random_forest"
        / "test_predictions.npy",
        "truth": FINAL_BASELINES_DIR
        / "random_forest"
        / "test_truth.npy",
        "mask": FINAL_BASELINES_DIR
        / "random_forest"
        / "test_mask.npy",
    },
    "lstm": {
        "directory": FINAL_BASELINES_DIR / "lstm",
        "predictions": FINAL_BASELINES_DIR
        / "lstm"
        / "test_predictions.npy",
        "truth": FINAL_BASELINES_DIR
        / "lstm"
        / "test_truth.npy",
        "mask": FINAL_BASELINES_DIR
        / "lstm"
        / "test_mask.npy",
    },
    "tcn": {
        "directory": FINAL_BASELINES_DIR / "tcn",
        "predictions": FINAL_BASELINES_DIR
        / "tcn"
        / "test_predictions.npy",
        "truth": FINAL_BASELINES_DIR
        / "tcn"
        / "test_truth.npy",
        "mask": FINAL_BASELINES_DIR
        / "tcn"
        / "test_mask.npy",
    },
    "transformer": {
        "directory": FINAL_TRANSFORMER_DIR,
        "predictions": FINAL_TRANSFORMER_DIR
        / "test_predictions_real_units.npy",
        "truth": FINAL_TRANSFORMER_DIR
        / "test_truth_real_units.npy",
        "mask": FINAL_TRANSFORMER_DIR
        / "test_mask.npy",
    },
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_plot_module_available():
    """
    Import the shared plotting implementation.

    Keeping plotting logic in evaluation_plots.py makes this script purely
    responsible for locating frozen evaluation artifacts and packaging plots.
    """
    baselines_dir = SINGLE_SITE_DIR / "baselines"

    if str(baselines_dir) not in sys.path:
        sys.path.insert(0, str(baselines_dir))

    try:
        from evaluation_plots import generate_evaluation_plots
    except ImportError as exc:
        raise RuntimeError(
            "Could not import baselines/evaluation_plots.py. "
            "Make sure the shared plotting utility exists."
        ) from exc

    return generate_evaluation_plots


def _load_array(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {name} file does not exist:\n{path}"
        )

    try:
        array = np.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {name}:\n{path}"
        ) from exc

    return np.asarray(array)


def _validate_arrays(
    model_name: str,
    predictions: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if predictions.ndim != 3:
        raise ValueError(
            f"{model_name}: predictions must have shape "
            f"(samples, horizons, targets), got {predictions.shape}"
        )

    if truth.ndim != 3:
        raise ValueError(
            f"{model_name}: truth must have shape "
            f"(samples, horizons, targets), got {truth.shape}"
        )

    if mask.ndim != 3:
        raise ValueError(
            f"{model_name}: mask must have shape "
            f"(samples, horizons, targets), got {mask.shape}"
        )

    if predictions.shape != truth.shape:
        raise ValueError(
            f"{model_name}: predictions and truth shapes differ: "
            f"{predictions.shape} vs {truth.shape}"
        )

    if predictions.shape != mask.shape:
        raise ValueError(
            f"{model_name}: predictions and mask shapes differ: "
            f"{predictions.shape} vs {mask.shape}"
        )

    if predictions.shape[1] != 6:
        raise ValueError(
            f"{model_name}: expected 6 forecast horizons, "
            f"got {predictions.shape[1]}"
        )

    if predictions.shape[2] != 2:
        raise ValueError(
            f"{model_name}: expected 2 targets (O3, NO2), "
            f"got {predictions.shape[2]}"
        )

    predictions = predictions.astype(float, copy=False)
    truth = truth.astype(float, copy=False)

    # Convert masks to boolean while preserving the evaluator's valid
    # locations. Any non-zero mask entry is treated as valid.
    mask = mask.astype(bool, copy=False)

    # Ensure non-finite values cannot enter plotting calculations.
    finite = (
        np.isfinite(predictions)
        & np.isfinite(truth)
    )

    mask = mask & finite

    return predictions, truth, mask


def _copy_training_curve(
    model_name: str,
    output_directory: Path,
) -> str | None:
    """
    Copy an already-existing training curve for sequence models.

    This does NOT create a training curve for XGBoost or Random Forest.
    We only expose curves that genuinely exist as training artifacts.
    """

    source_candidates = []

    if model_name == "lstm":
        source_candidates = [
            SINGLE_SITE_DIR
            / "outputs"
            / "plots"
            / "lstm_loss_curve.png",
            FINAL_BASELINES_DIR
            / "lstm"
            / "loss_curve.png",
        ]

    elif model_name == "tcn":
        source_candidates = [
            SINGLE_SITE_DIR
            / "outputs"
            / "plots"
            / "tcn_loss_curve.png",
            FINAL_BASELINES_DIR
            / "tcn"
            / "loss_curve.png",
        ]

    elif model_name == "transformer":
        source_candidates = [
            FINAL_TRANSFORMER_DIR / "loss_curve.png",
            SINGLE_SITE_DIR
            / "outputs"
            / "plots"
            / "loss_curve.png",
        ]

    else:
        return None

    for source in source_candidates:
        if source.exists():
            destination = (
                output_directory
                / f"{model_name}_training_loss.png"
            )

            shutil.copy2(source, destination)
            return str(destination)

    return None


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_model_plots(model_name: str) -> Dict[str, str]:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {', '.join(MODEL_CONFIGS)}"
        )

    config = MODEL_CONFIGS[model_name]

    predictions = _load_array(
        config["predictions"],
        "predictions",
    )

    truth = _load_array(
        config["truth"],
        "truth",
    )

    mask = _load_array(
        config["mask"],
        "mask",
    )

    predictions, truth, mask = _validate_arrays(
        model_name,
        predictions,
        truth,
        mask,
    )

    output_directory = PLOTS_ROOT / model_name
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generate_evaluation_plots = _ensure_plot_module_available()

    generated = generate_evaluation_plots(
        model_name=model_name,
        predictions=predictions,
        truth=truth,
        mask=mask,
        output_dir=output_directory,
    )

    training_curve = _copy_training_curve(
        model_name=model_name,
        output_directory=output_directory,
    )

    if training_curve is not None:
        generated["training_loss"] = training_curve

    return generated


def generate_all_plots() -> Dict[str, Dict[str, str]]:
    PLOTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: Dict[str, Dict[str, str]] = {}

    for model_name in MODEL_CONFIGS:
        print(
            f"[StratoWatch] Generating evaluation plots: "
            f"{model_name}"
        )

        try:
            generated = generate_model_plots(model_name)

            results[model_name] = generated

            print(
                f"[StratoWatch] Generated "
                f"{len(generated)} plots for {model_name}"
            )

        except Exception as exc:
            print(
                f"[StratoWatch] Plot generation failed for "
                f"{model_name}: {exc}"
            )

            # Do not hide failures from the caller.
            # The model evaluation itself remains untouched.
            raise

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        results = generate_all_plots()

        print()
        print("=" * 72)
        print("SINGLE-SITE EVALUATION PLOTS GENERATED")
        print("=" * 72)

        total = 0

        for model_name, plots in results.items():
            print(f"\n{model_name.upper()}")

            for plot_name, path in plots.items():
                print(f"  {plot_name}: {path}")

            total += len(plots)

        print()
        print(f"Total generated/copied plots: {total}")
        print(f"Output root: {PLOTS_ROOT}")
        print("=" * 72)

        return 0

    except Exception as exc:
        print()
        print("=" * 72)
        print("SINGLE-SITE PLOT GENERATION FAILED")
        print("=" * 72)
        print(str(exc))
        print("=" * 72)

        return 1


if __name__ == "__main__":
    raise SystemExit(main())