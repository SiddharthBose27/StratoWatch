from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# STRATOWATCH 2.0
# FINAL RESEARCH PLOTS
#
# IMPORTANT:
# - No model training
# - No model evaluation
# - No retraining
# - Uses frozen final results only
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()

MULTI_SITE_ROOT = THIS_FILE.parents[1]
BACKEND_ROOT = MULTI_SITE_ROOT.parent

SINGLE_SITE_ROOT = BACKEND_ROOT / "stratowatch_single_site"

SINGLE_RESULTS_PATH = (
    SINGLE_SITE_ROOT
    / "outputs"
    / "final_baselines"
    / "final_single_site_results.json"
)

MULTI_EVAL_DIR = (
    MULTI_SITE_ROOT
    / "outputs"
    / "final_evaluation"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "research_results"
    / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# FROZEN MULTI-SITE RESULTS
#
# These are the already verified final test results.
# They are NOT recomputed here.
# ------------------------------------------------------------

MULTI_RESULTS = {
    "st_transformer": {
        "label": "ST Transformer",
        "mae": 18.8349,
        "rmse": 27.9747,
        "r2": 0.3437,
    },

    "static_graph_st": {
        "label": "Static Graph-ST",
        "mae": 19.038433,
        "rmse": 27.078136,
        "r2": 0.385048,
    },

    "dynamic_wind_graph_st": {
        "label": "Dynamic Wind Graph-ST",
        "mae": 19.046450,
        "rmse": 26.945461,
        "r2": 0.391060,
    },
}


MULTI_TARGET_RESULTS = {
    "static_graph_st": {
        "O3": {
            "mae": 18.693880,
            "rmse": 27.554443,
            "r2": 0.474189,
        },
        "NO2": {
            "mae": 19.382982,
            "rmse": 26.593300,
            "r2": 0.210967,
        },
    },

    "dynamic_wind_graph_st": {
        "O3": {
            "mae": 18.785820,
            "rmse": 27.483654,
            "r2": 0.476888,
        },
        "NO2": {
            "mae": 19.307077,
            "rmse": 26.396299,
            "r2": 0.222614,
        },
    },

    "st_transformer": {
        "O3": {
            "mae": 19.0729,
            "rmse": 29.2425,
            "r2": 0.4078,
        },
        "NO2": {
            "mae": 18.5969,
            "rmse": 26.6466,
            "r2": 0.2078,
        },
    },
}


MULTI_HORIZON_RESULTS = {
    "static_graph_st": {
        "mae": [
            18.934742,
            18.947809,
            18.946615,
            19.083321,
            19.127953,
            19.190237,
        ],
        "rmse": [
            27.033979,
            27.055622,
            26.832195,
            27.078140,
            27.100542,
            27.365721,
        ],
    },

    "dynamic_wind_graph_st": {
        "mae": [
            18.856054,
            18.959045,
            18.996576,
            19.089285,
            19.141378,
            19.236473,
        ],
        "rmse": [
            26.796251,
            26.858864,
            26.705662,
            26.995502,
            27.012739,
            27.299803,
        ],
    },
}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_plot(filename):
    path = OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  ✓ {filename}")


def get_single_model(results, model_name):

    for record in results["models"]:

        if record["model"] == model_name:
            return record

    raise KeyError(
        f"Single-site model '{model_name}' not found."
    )


# ============================================================
# LOAD SINGLE-SITE RESULTS
# ============================================================

print("=" * 70)
print("STRATOWATCH 2.0 — FINAL RESEARCH PLOTS")
print("=" * 70)

print("\nLoading frozen single-site results...")

if not SINGLE_RESULTS_PATH.exists():

    raise FileNotFoundError(
        f"Single-site results not found:\n"
        f"{SINGLE_RESULTS_PATH}"
    )


single_results = load_json(
    SINGLE_RESULTS_PATH
)

print("✓ Single-site results loaded")

print("✓ Multi-site frozen results loaded")


# ============================================================
# SINGLE-SITE MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("SINGLE-SITE FIGURES")
print("=" * 70)


SINGLE_MODELS = [
    ("xgboost", "XGBoost"),
    ("random_forest", "Random Forest"),
    ("tcn", "TCN"),
    ("transformer", "Transformer"),
    ("lstm", "LSTM"),
    ("forecast_only", "Forecast-only"),
]


single_names = [
    label
    for _, label in SINGLE_MODELS
]


single_mae = []
single_rmse = []
single_r2 = []


for model_key, _ in SINGLE_MODELS:

    record = get_single_model(
        single_results,
        model_key,
    )

    single_mae.append(
        record["MAE"]
    )

    single_rmse.append(
        record["RMSE"]
    )

    single_r2.append(
        record["R2"]
    )


# ------------------------------------------------------------
# Figure 1 — Single-site MAE
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    single_names,
    single_mae,
)

plt.ylabel("MAE")
plt.title(
    "Single-Site Model Comparison — MAE"
)

plt.xticks(
    rotation=20,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.25,
)

save_plot(
    "01_single_site_mae.png"
)


# ------------------------------------------------------------
# Figure 2 — Single-site RMSE
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    single_names,
    single_rmse,
)

plt.ylabel("RMSE")
plt.title(
    "Single-Site Model Comparison — RMSE"
)

plt.xticks(
    rotation=20,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.25,
)

save_plot(
    "02_single_site_rmse.png"
)


# ------------------------------------------------------------
# Figure 3 — Single-site R²
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    single_names,
    single_r2,
)

plt.ylabel("R²")
plt.title(
    "Single-Site Model Comparison — R²"
)

plt.xticks(
    rotation=20,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.25,
)

save_plot(
    "03_single_site_r2.png"
)


# ============================================================
# SINGLE-SITE O3 vs NO2
# ============================================================

target_metrics = single_results[
    "target_metrics"
]


o3_mae = []
no2_mae = []


for model_key, _ in SINGLE_MODELS:

    o3_mae.append(
        target_metrics[
            model_key
        ]["O3"]["MAE"]
    )

    no2_mae.append(
        target_metrics[
            model_key
        ]["NO2"]["MAE"]
    )


x = np.arange(
    len(single_names)
)

width = 0.38


# ------------------------------------------------------------
# Figure 4 — Single-site target MAE
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    o3_mae,
    width,
    label="O3",
)

plt.bar(
    x + width / 2,
    no2_mae,
    width,
    label="NO2",
)

plt.xticks(
    x,
    single_names,
    rotation=20,
    ha="right",
)

plt.ylabel("MAE")

plt.title(
    "Single-Site Target-wise MAE"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25,
)

save_plot(
    "04_single_site_o3_no2_mae.png"
)


# ============================================================
# SINGLE-SITE HORIZON PERFORMANCE
# ============================================================

horizon_wise = single_results[
    "horizon_wise"
]


# ------------------------------------------------------------
# Figure 5 — XGBoost vs Transformer horizon MAE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))


for model_key, label in [
    ("xgboost", "XGBoost"),
    ("transformer", "Transformer"),
]:

    values = []

    for h in range(1, 7):

        data = horizon_wise[
            model_key
        ]

        if f"H+{h}" in data:

            key = f"H+{h}"

        else:

            key = f"horizon_{h}"

        values.append(
            data[key]["MAE"]
        )


    plt.plot(
        range(1, 7),
        values,
        marker="o",
        linewidth=2,
        label=label,
    )


plt.xlabel(
    "Forecast Horizon"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Single-Site Horizon-wise MAE"
)

plt.xticks(
    range(1, 7),
    [
        "H+1",
        "H+2",
        "H+3",
        "H+4",
        "H+5",
        "H+6",
    ],
)

plt.legend()

plt.grid(
    alpha=0.25
)

save_plot(
    "05_single_site_horizon_mae.png"
)


# ============================================================
# MULTI-SITE MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MULTI-SITE FIGURES")
print("=" * 70)


multi_names = [
    MULTI_RESULTS[
        "st_transformer"
    ]["label"],

    MULTI_RESULTS[
        "static_graph_st"
    ]["label"],

    MULTI_RESULTS[
        "dynamic_wind_graph_st"
    ]["label"],
]


multi_mae = [
    MULTI_RESULTS[
        "st_transformer"
    ]["mae"],

    MULTI_RESULTS[
        "static_graph_st"
    ]["mae"],

    MULTI_RESULTS[
        "dynamic_wind_graph_st"
    ]["mae"],
]


multi_rmse = [
    MULTI_RESULTS[
        "st_transformer"
    ]["rmse"],

    MULTI_RESULTS[
        "static_graph_st"
    ]["rmse"],

    MULTI_RESULTS[
        "dynamic_wind_graph_st"
    ]["rmse"],
]


multi_r2 = [
    MULTI_RESULTS[
        "st_transformer"
    ]["r2"],

    MULTI_RESULTS[
        "static_graph_st"
    ]["r2"],

    MULTI_RESULTS[
        "dynamic_wind_graph_st"
    ]["r2"],
]


# ------------------------------------------------------------
# Figure 6 — Multi-site MAE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    multi_names,
    multi_mae,
)

plt.ylabel("MAE")

plt.title(
    "Multi-Site Model Comparison — MAE"
)

plt.xticks(
    rotation=15,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.25,
)

save_plot(
    "06_multi_site_mae.png"
)


# ------------------------------------------------------------
# Figure 7 — Multi-site RMSE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    multi_names,
    multi_rmse,
)

plt.ylabel("RMSE")

plt.title(
    "Multi-Site Model Comparison — RMSE"
)

plt.xticks(
    rotation=15,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.25,
)

save_plot(
    "07_multi_site_rmse.png"
)


# ------------------------------------------------------------
# Figure 8 — Multi-site R²
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    multi_names,
    multi_r2,
)

plt.ylabel("R²")

plt.title(
    "Multi-Site Model Comparison — R²"
)

plt.xticks(
    rotation=15,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.25,
)

save_plot(
    "08_multi_site_r2.png"
)


# ============================================================
# MULTI-SITE O3 vs NO2
# ============================================================


multi_o3 = [
    MULTI_TARGET_RESULTS[
        "st_transformer"
    ]["O3"]["mae"],

    MULTI_TARGET_RESULTS[
        "static_graph_st"
    ]["O3"]["mae"],

    MULTI_TARGET_RESULTS[
        "dynamic_wind_graph_st"
    ]["O3"]["mae"],
]


multi_no2 = [
    MULTI_TARGET_RESULTS[
        "st_transformer"
    ]["NO2"]["mae"],

    MULTI_TARGET_RESULTS[
        "static_graph_st"
    ]["NO2"]["mae"],

    MULTI_TARGET_RESULTS[
        "dynamic_wind_graph_st"
    ]["NO2"]["mae"],
]


x = np.arange(
    len(multi_names)
)


# ------------------------------------------------------------
# Figure 9 — Multi-site target MAE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    multi_o3,
    width,
    label="O3",
)

plt.bar(
    x + width / 2,
    multi_no2,
    width,
    label="NO2",
)

plt.xticks(
    x,
    multi_names,
    rotation=15,
    ha="right",
)

plt.ylabel("MAE")

plt.title(
    "Multi-Site Target-wise MAE"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25,
)

save_plot(
    "09_multi_site_o3_no2_mae.png"
)


# ============================================================
# MULTI-SITE HORIZON PERFORMANCE
# ============================================================


# ------------------------------------------------------------
# Figure 10 — Graph models horizon MAE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))


for model_key, label in [
    (
        "static_graph_st",
        "Static Graph-ST",
    ),
    (
        "dynamic_wind_graph_st",
        "Dynamic Wind Graph-ST",
    ),
]:

    values = MULTI_HORIZON_RESULTS[
        model_key
    ]["mae"]


    plt.plot(
        range(1, 7),
        values,
        marker="o",
        linewidth=2,
        label=label,
    )


plt.xlabel(
    "Forecast Horizon"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Multi-Site Graph Models — Horizon-wise MAE"
)

plt.xticks(
    range(1, 7),
    [
        "H+1",
        "H+2",
        "H+3",
        "H+4",
        "H+5",
        "H+6",
    ],
)

plt.legend()

plt.grid(
    alpha=0.25
)

save_plot(
    "10_multi_site_horizon_mae.png"
)


# ------------------------------------------------------------
# Figure 11 — Graph models horizon RMSE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))


for model_key, label in [
    (
        "static_graph_st",
        "Static Graph-ST",
    ),
    (
        "dynamic_wind_graph_st",
        "Dynamic Wind Graph-ST",
    ),
]:

    values = MULTI_HORIZON_RESULTS[
        model_key
    ]["rmse"]


    plt.plot(
        range(1, 7),
        values,
        marker="o",
        linewidth=2,
        label=label,
    )


plt.xlabel(
    "Forecast Horizon"
)

plt.ylabel(
    "RMSE"
)

plt.title(
    "Multi-Site Graph Models — Horizon-wise RMSE"
)

plt.xticks(
    range(1, 7),
    [
        "H+1",
        "H+2",
        "H+3",
        "H+4",
        "H+5",
        "H+6",
    ],
)

plt.legend()

plt.grid(
    alpha=0.25
)

save_plot(
    "11_multi_site_horizon_rmse.png"
)


# ============================================================
# PREDICTION VS ACTUAL
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION VISUALIZATION")
print("=" * 70)


prediction_path = (
    MULTI_EVAL_DIR
    / "dynamic_test_predictions_real.npy"
)

truth_path = (
    MULTI_EVAL_DIR
    / "dynamic_test_truth_real.npy"
)

mask_path = (
    MULTI_EVAL_DIR
    / "dynamic_test_mask.npy"
)


if (
    prediction_path.exists()
    and truth_path.exists()
    and mask_path.exists()
):

    predictions = np.load(
        prediction_path
    )

    truth = np.load(
        truth_path
    )

    mask = np.load(
        mask_path
    ).astype(bool)


    print(
        f"Prediction array: {predictions.shape}"
    )

    print(
        f"Truth array:       {truth.shape}"
    )

    print(
        f"Mask array:        {mask.shape}"
    )


    # H+1 prediction vs actual.
    # This is only for visualization.
    # It does not change evaluation metrics.

    for target_idx, target_name in [
        (0, "O3"),
        (1, "NO2"),
    ]:

        valid = mask[
            :,
            0,
            :,
            target_idx,
        ]


        actual = truth[
            :,
            0,
            :,
            target_idx,
        ][valid]


        predicted = predictions[
            :,
            0,
            :,
            target_idx,
        ][valid]


        # Limit visual points only.
        n = min(
            len(actual),
            20000,
        )


        actual = actual[:n]
        predicted = predicted[:n]


        plt.figure(
            figsize=(7, 7)
        )


        plt.scatter(
            actual,
            predicted,
            alpha=0.15,
            s=8,
        )


        low = min(
            actual.min(),
            predicted.min(),
        )

        high = max(
            actual.max(),
            predicted.max(),
        )


        plt.plot(
            [low, high],
            [low, high],
            linestyle="--",
            linewidth=1.5,
        )


        plt.xlabel(
            f"Actual {target_name}"
        )

        plt.ylabel(
            f"Predicted {target_name}"
        )

        plt.title(
            f"Dynamic Wind Graph-ST — "
            f"{target_name} — H+1"
        )

        plt.grid(
            alpha=0.25
        )


        save_plot(
            f"12_prediction_vs_actual_{target_name.lower()}.png"
        )


else:

    print(
        "⚠ Dynamic prediction arrays not found."
    )

    print(
        "  Prediction-vs-actual plots skipped."
    )


# ============================================================
# MANIFEST
# ============================================================

figure_files = sorted(
    p.name
    for p in OUTPUT_DIR.glob(
        "*.png"
    )
)


manifest = {
    "project": "StratoWatch 2.0",

    "purpose": (
        "Final research figures "
        "from frozen evaluation artifacts"
    ),

    "training_performed": False,

    "evaluation_performed": False,

    "single_site_source": str(
        SINGLE_RESULTS_PATH
    ),

    "multi_site_source": (
        "Frozen final multi-site "
        "evaluation results"
    ),

    "figures": figure_files,
}


with open(
    OUTPUT_DIR / "plot_manifest.json",
    "w",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("FINAL RESEARCH PLOTS COMPLETE")
print("=" * 70)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    f"\nFigures created: {len(figure_files)}"
)

print(
    "\nTraining performed: NO"
)

print(
    "Evaluation rerun:   NO"
)

print(
    "\nAll figures use frozen StratoWatch results."
)

print("=" * 70)