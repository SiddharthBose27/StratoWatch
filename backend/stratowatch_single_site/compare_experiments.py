import pandas as pd

full = pd.read_csv("experiments/full_model/metrics_summary.csv").iloc[0]
nosat = pd.read_csv("experiments/no_satellite/metrics_summary.csv").iloc[0]
tuned = pd.read_csv("experiments/tuned_model/metrics_summary.csv").iloc[0]

rows = [
    ["Forecast-only Baseline", full["baseline_mae"], full["baseline_rmse"], full["baseline_r2"]],
    ["Full Model (Residual + Satellite)", full["model_mae"], full["model_rmse"], full["model_r2"]],
    ["Ablation: No Satellite", nosat["model_mae"], nosat["model_rmse"], nosat["model_r2"]],
    ["Tuned Transformer (128/8/3)", tuned["model_mae"], tuned["model_rmse"], tuned["model_r2"]],
]
df = pd.DataFrame(rows, columns=["Method", "MAE", "RMSE", "R2"])
print(df.to_string(index=False))
df.to_csv("outputs/final_comparison_table.csv", index=False)
print("\n✅ Saved: outputs/final_comparison_table.csv")