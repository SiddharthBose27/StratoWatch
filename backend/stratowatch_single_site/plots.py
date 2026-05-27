# plots.py
# Purpose:
#   Generate paper-ready plots from evaluation results

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load predictions
# -------------------------
df = pd.read_csv("outputs/predictions_test.csv")

os.makedirs("outputs/plots", exist_ok=True)

print("✅ Loaded predictions:", df.shape)

# -------------------------
# 1️⃣ True vs Predicted (O3)
# -------------------------
plt.figure(figsize=(12,5))
plt.plot(df["true_O3"].values[:500], label="True O3", linewidth=2)
plt.plot(df["pred_O3"].values[:500], label="Pred O3", linewidth=2)
plt.title("True vs Predicted O3 (First 500 Samples)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/o3_true_vs_pred.png", dpi=300)
plt.close()

# -------------------------
# 2️⃣ True vs Predicted (NO2)
# -------------------------
plt.figure(figsize=(12,5))
plt.plot(df["true_NO2"].values[:500], label="True NO2", linewidth=2)
plt.plot(df["pred_NO2"].values[:500], label="Pred NO2", linewidth=2)
plt.title("True vs Predicted NO2 (First 500 Samples)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/no2_true_vs_pred.png", dpi=300)
plt.close()

# -------------------------
# 3️⃣ Scatter Plot (O3)
# -------------------------
plt.figure(figsize=(6,6))
sns.scatterplot(x="true_O3", y="pred_O3", data=df, alpha=0.4)
plt.plot([df["true_O3"].min(), df["true_O3"].max()],
         [df["true_O3"].min(), df["true_O3"].max()],
         color="red", linestyle="--")
plt.title("O3: True vs Predicted Scatter")
plt.tight_layout()
plt.savefig("outputs/plots/o3_scatter.png", dpi=300)
plt.close()

# -------------------------
# 4️⃣ Scatter Plot (NO2)
# -------------------------
plt.figure(figsize=(6,6))
sns.scatterplot(x="true_NO2", y="pred_NO2", data=df, alpha=0.4)
plt.plot([df["true_NO2"].min(), df["true_NO2"].max()],
         [df["true_NO2"].min(), df["true_NO2"].max()],
         color="red", linestyle="--")
plt.title("NO2: True vs Predicted Scatter")
plt.tight_layout()
plt.savefig("outputs/plots/no2_scatter.png", dpi=300)
plt.close()

# -------------------------
# 5️⃣ Residual Distribution
# -------------------------
df["residual_O3"] = df["true_O3"] - df["pred_O3"]
df["residual_NO2"] = df["true_NO2"] - df["pred_NO2"]

plt.figure(figsize=(10,5))
sns.histplot(df["residual_O3"], bins=50, kde=True)
plt.title("Residual Distribution - O3")
plt.tight_layout()
plt.savefig("outputs/plots/o3_residual_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(10,5))
sns.histplot(df["residual_NO2"], bins=50, kde=True)
plt.title("Residual Distribution - NO2")
plt.tight_layout()
plt.savefig("outputs/plots/no2_residual_distribution.png", dpi=300)
plt.close()

print("✅ All plots saved to outputs/plots/")