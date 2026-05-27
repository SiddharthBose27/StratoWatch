import pandas as pd

m = pd.read_csv("outputs/metrics_summary.csv")

table = pd.DataFrame([
    ["Forecast-only baseline", m.loc[0,"baseline_mae"], m.loc[0,"baseline_rmse"], m.loc[0,"baseline_r2"]],
    ["Residual Transformer (y = forecast + Δ)", m.loc[0,"model_mae"], m.loc[0,"model_rmse"], m.loc[0,"model_r2"]],
], columns=["Method", "MAE", "RMSE", "R2"])

print("\n✅ Paper Table (copy into report):\n")
print(table.to_string(index=False))

table.to_csv("outputs/paper_results_table.csv", index=False)
print("\n✅ Saved: outputs/paper_results_table.csv")