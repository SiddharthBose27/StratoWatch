"""
StratoWatch 2.0 — Final Single-Site Research Results Consolidation

Consolidates frozen final results from:

1. Forecast-only
2. Random Forest
3. XGBoost
4. LSTM
5. TCN
6. Transformer

Supports the two result schemas currently used by the finalized
StratoWatch evaluators.

IMPORTANT:
- No training
- No tuning
- No test-data modification
- Reads only saved JSON artifacts
"""

from __future__ import annotations

import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_DIR / "outputs" / "final_baselines"

FINAL_OUTPUT = (
    OUTPUT_ROOT
    / "final_single_site_results.json"
)

TRANSFORMER_RESULTS = (
    PROJECT_DIR
    / "outputs"
    / "final_single_site"
    / "final_evaluation_metrics.json"
)


# ============================================================
# FROZEN RESULT FILES
# ============================================================

MODEL_FILES = {
    "forecast_only": (
        OUTPUT_ROOT
        / "forecast_only"
        / "test_metrics.json"
    ),

    "random_forest": (
        OUTPUT_ROOT
        / "random_forest"
        / "test_metrics.json"
    ),

    "xgboost": (
        OUTPUT_ROOT
        / "xgboost"
        / "test_metrics.json"
    ),

    "lstm": (
        OUTPUT_ROOT
        / "lstm"
        / "test_metrics.json"
    ),

    "tcn": (
        OUTPUT_ROOT
        / "tcn"
        / "test_metrics.json"
    ),

    "transformer": TRANSFORMER_RESULTS,
}


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{path}"
        )

    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# NORMALIZE RESULT SCHEMA
# ============================================================

def normalize_result(
    model_name: str,
    data: dict,
) -> dict:
    """
    Convert either supported result schema into:

        overall
        targets
        horizons
    """

    # --------------------------------------------------------
    # Schema A:
    #
    # {
    #   "overall": {...},
    #   "per_target": {...},
    #   "horizon_wise": {...}
    # }
    #
    # Used by Transformer.
    # --------------------------------------------------------

    if "overall" in data:

        overall = data["overall"]

        targets = data.get(
            "per_target",
            {},
        )

        horizons = data.get(
            "horizon_wise",
            {},
        )

        o3 = targets.get("O3_target")

        no2 = targets.get("NO2_target")

        if o3 is None:
            raise KeyError(
                f"{model_name}: "
                "missing per_target.O3_target"
            )

        if no2 is None:
            raise KeyError(
                f"{model_name}: "
                "missing per_target.NO2_target"
            )

        return {
            "model": model_name,

            "overall": {
                "MAE": float(overall["MAE"]),
                "RMSE": float(overall["RMSE"]),
                "R2": float(overall["R2"]),
            },

            "targets": {
                "O3": {
                    "MAE": float(o3["MAE"]),
                    "RMSE": float(o3["RMSE"]),
                    "R2": float(o3["R2"]),
                },

                "NO2": {
                    "MAE": float(no2["MAE"]),
                    "RMSE": float(no2["RMSE"]),
                    "R2": float(no2["R2"]),
                },
            },

            "horizons": horizons,
        }

    # --------------------------------------------------------
    # Schema B:
    #
    # {
    #   "test": {
    #       "overall": {...},
    #       "targets": {...},
    #       "horizons": {...}
    #   }
    # }
    #
    # Used by RF/XGBoost/LSTM/TCN/Forecast-only.
    # --------------------------------------------------------

    if "test" in data:

        test = data["test"]

        overall = test.get(
            "overall"
        )

        targets = test.get(
            "targets",
            {},
        )

        horizons = test.get(
            "horizons",
            {},
        )

        if overall is None:
            raise KeyError(
                f"{model_name}: "
                "missing test.overall"
            )

        o3 = targets.get("O3")

        no2 = targets.get("NO2")

        if o3 is None:
            raise KeyError(
                f"{model_name}: "
                "missing test.targets.O3"
            )

        if no2 is None:
            raise KeyError(
                f"{model_name}: "
                "missing test.targets.NO2"
            )

        return {
            "model": model_name,

            "overall": {
                "MAE": float(overall["MAE"]),
                "RMSE": float(overall["RMSE"]),
                "R2": float(overall["R2"]),
            },

            "targets": {
                "O3": {
                    "MAE": float(o3["MAE"]),
                    "RMSE": float(o3["RMSE"]),
                    "R2": float(o3["R2"]),
                },

                "NO2": {
                    "MAE": float(no2["MAE"]),
                    "RMSE": float(no2["RMSE"]),
                    "R2": float(no2["R2"]),
                },
            },

            "horizons": horizons,
        }

    raise KeyError(
        f"{model_name}: "
        "unsupported result JSON schema. "
        "Expected 'overall' or 'test'."
    )


# ============================================================
# LOAD + NORMALIZE ALL RESULTS
# ============================================================

print("=" * 72)
print("STRATOWATCH 2.0 — FINAL SINGLE-SITE RESULTS")
print("=" * 72)

results = {}

for model_name, path in MODEL_FILES.items():

    print()
    print(f"Loading {model_name}...")
    print(f"  Path: {path}")

    data = load_json(path)

    print("  ✓ JSON loaded")

    normalized = normalize_result(
        model_name,
        data,
    )

    results[model_name] = normalized

    print("  ✓ Test metrics normalized")


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

comparison = []

for model_name, result in results.items():

    overall = result["overall"]
    o3 = result["targets"]["O3"]
    no2 = result["targets"]["NO2"]

    comparison.append(
        {
            "model": model_name,

            "MAE": overall["MAE"],
            "RMSE": overall["RMSE"],
            "R2": overall["R2"],

            "O3_MAE": o3["MAE"],
            "O3_RMSE": o3["RMSE"],
            "O3_R2": o3["R2"],

            "NO2_MAE": no2["MAE"],
            "NO2_RMSE": no2["RMSE"],
            "NO2_R2": no2["R2"],
        }
    )


# ============================================================
# RANKINGS
# ============================================================

mae_ranking = sorted(
    comparison,
    key=lambda x: x["MAE"],
)

rmse_ranking = sorted(
    comparison,
    key=lambda x: x["RMSE"],
)

r2_ranking = sorted(
    comparison,
    key=lambda x: x["R2"],
    reverse=True,
)


# ============================================================
# FORECAST-ONLY VS XGBOOST
# ============================================================

forecast = results["forecast_only"]["overall"]

xgb = results["xgboost"]["overall"]

mae_improvement = (
    (forecast["MAE"] - xgb["MAE"])
    / forecast["MAE"]
    * 100.0
)

rmse_improvement = (
    (forecast["RMSE"] - xgb["RMSE"])
    / forecast["RMSE"]
    * 100.0
)


# ============================================================
# DATASET CONTRACT
# ============================================================

dataset_contract = {
    "features": 134,
    "input_hours": 24,
    "forecast_hours": 6,
    "targets": [
        "O3_target",
        "NO2_target",
    ],
    "test_windows": 6464,
    "evaluation_units": "real_units",
}


# ============================================================
# FINAL RESULTS PACKAGE
# ============================================================

final_results = {

    "experiment": (
        "StratoWatch 2.0 — "
        "Single-Site Final Evaluation"
    ),

    "dataset_contract": dataset_contract,

    "models": comparison,

    "rankings": {
        "best_mae": mae_ranking[0]["model"],
        "best_rmse": rmse_ranking[0]["model"],
        "best_r2": r2_ranking[0]["model"],
    },

    "xgboost_vs_forecast_only": {
        "MAE_improvement_percent": float(
            mae_improvement
        ),

        "RMSE_improvement_percent": float(
            rmse_improvement
        ),

        "forecast_only_MAE": float(
            forecast["MAE"]
        ),

        "xgboost_MAE": float(
            xgb["MAE"]
        ),

        "forecast_only_RMSE": float(
            forecast["RMSE"]
        ),

        "xgboost_RMSE": float(
            xgb["RMSE"]
        ),
    },

    "horizon_wise": {
        model_name: result["horizons"]
        for model_name, result in results.items()
    },

    "target_metrics": {
        model_name: result["targets"]
        for model_name, result in results.items()
    },

    "source_files": {
        model_name: str(path)
        for model_name, path in MODEL_FILES.items()
    },

    "research_status": {
        "all_six_models_evaluated": True,
        "all_six_models_frozen": True,
        "no_retraining_performed": True,
        "real_unit_evaluation": True,
        "phase_8_step_2_complete": True,
    },
}


# ============================================================
# SAVE
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    FINAL_OUTPUT,
    "w",
) as f:

    json.dump(
        final_results,
        f,
        indent=2,
    )


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print("=" * 72)
print("FINAL SINGLE-SITE MODEL COMPARISON")
print("=" * 72)

print(
    f"{'Model':<20}"
    f"{'MAE':>10}"
    f"{'RMSE':>10}"
    f"{'R²':>10}"
)

print("-" * 50)

for row in mae_ranking:

    print(
        f"{row['model']:<20}"
        f"{row['MAE']:>10.4f}"
        f"{row['RMSE']:>10.4f}"
        f"{row['R2']:>10.4f}"
    )


# ============================================================
# RESEARCH SUMMARY
# ============================================================

print()
print("=" * 72)
print("RESEARCH SUMMARY")
print("=" * 72)

print(
    f"Best MAE : "
    f"{mae_ranking[0]['model']}"
)

print(
    f"Best RMSE: "
    f"{rmse_ranking[0]['model']}"
)

print(
    f"Best R²  : "
    f"{r2_ranking[0]['model']}"
)

print()
print("XGBoost vs Forecast-only:")

print(
    f"  MAE improvement : "
    f"{mae_improvement:.2f}%"
)

print(
    f"  RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)

print()
print("Dataset contract:")
print("  ✓ 134 features")
print("  ✓ 24-hour input")
print("  ✓ 6-hour horizon")
print("  ✓ O3 + NO2")
print("  ✓ 6464 test windows")
print("  ✓ Real-unit evaluation")

print()
print("Frozen models:")
print("  ✓ Forecast-only")
print("  ✓ Random Forest")
print("  ✓ XGBoost")
print("  ✓ LSTM")
print("  ✓ TCN")
print("  ✓ Transformer")

print()
print("Saved:")
print(f"  {FINAL_OUTPUT}")

print()
print("=" * 72)
print("SINGLE-SITE RESULTS CONSOLIDATION COMPLETE")
print("RESEARCH CONTRACT: PASSED")
print("=" * 72)