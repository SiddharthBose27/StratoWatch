# evaluate.py
# Purpose:
#   Evaluate trained residual-learning Transformer:
#   (1) predicts residuals in scaled space
#   (2) inverse-scales residuals to real units
#   (3) adds forecast back to get final targets
#   (4) compares against forecast-only baseline
#   (5) saves CSVs for paper plots

import os, json
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.temporal_transformer import TemporalTransformer


# =========================
# CONFIG
# =========================
DATA_PATH = "data/sequences.npz"
CKPT_PATH = "outputs/checkpoints/best_model.pt"
FEATURE_LIST_PATH = "data/feature_list.json"
Y_RES_SCALER_PATH = "data/y_res_scaler.pkl"

# Must match train.py hyperparams
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3

# Must match train.py splitting ratios
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 256


# =========================
# DEVICE (Mac M1)
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("✅ Using device:", device)


# =========================
# LOAD SEQUENCES
# =========================
data = np.load(DATA_PATH)
X_seq = data["X_seq"]   # (N, T, F)
y_seq = data["y_seq"]   # (N, 2)

N, T, F = X_seq.shape
print("✅ Loaded:", X_seq.shape, y_seq.shape)


# =========================
# SAME TIME SPLIT AS train.py
# =========================
test_size = int(N * TEST_RATIO)
val_size  = int(N * VAL_RATIO)
train_size = N - val_size - test_size

X_test = X_seq[train_size + val_size:]
y_test = y_seq[train_size + val_size:]

print("✅ Test set:", X_test.shape, y_test.shape)


# =========================
# LOAD MODEL + CHECKPOINT
# =========================
model = TemporalTransformer(
    in_dim=F,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    out_dim=2
).to(device)

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()
print("✅ Loaded checkpoint:", CKPT_PATH)


# =========================
# PREDICT (scaled residuals)
# =========================
X_test_t = torch.tensor(X_test, dtype=torch.float32)

preds_scaled = []
with torch.no_grad():
    for i in range(0, len(X_test_t), BATCH_SIZE):
        xb = X_test_t[i:i+BATCH_SIZE].to(device)
        pred = model(xb).cpu().numpy()
        preds_scaled.append(pred)

preds_scaled = np.vstack(preds_scaled)   # (N_test, 2)
trues_scaled = y_test                    # (N_test, 2)

print("✅ Predicted scaled residuals:", preds_scaled.shape)


# =========================
# INVERSE SCALE RESIDUALS (real units)
# =========================
y_res_scaler = joblib.load(Y_RES_SCALER_PATH)

preds_residual = y_res_scaler.inverse_transform(preds_scaled)
trues_residual = y_res_scaler.inverse_transform(trues_scaled)

print("✅ Residuals in real units:", preds_residual.shape)


# =========================
# GET FORECAST VALUES (from last step of input sequence)
# =========================
with open(FEATURE_LIST_PATH, "r") as f:
    feature_list = json.load(f)

o3_idx  = feature_list.index("O3_forecast")
no2_idx = feature_list.index("NO2_forecast")

forecast = np.stack([
    X_test[:, -1, o3_idx],
    X_test[:, -1, no2_idx]
], axis=1)  # (N_test, 2)

print("✅ Forecast extracted:", forecast.shape)


# =========================
# RECONSTRUCT FINAL TARGETS
# y = forecast + residual
# =========================
preds_real = forecast + preds_residual
trues_real = forecast + trues_residual


# =========================
# METRICS (MODEL)
# =========================
mae = mean_absolute_error(trues_real, preds_real)
mse = mean_squared_error(trues_real, preds_real)
rmse = np.sqrt(mse)
r2 = r2_score(trues_real, preds_real)

print("\n✅ MODEL (Residual + Forecast)")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")


# =========================
# BASELINE (forecast only)
# =========================
baseline_real = forecast

mae_b = mean_absolute_error(trues_real, baseline_real)
mse_b = mean_squared_error(trues_real, baseline_real)
rmse_b = np.sqrt(mse_b)
r2_b = r2_score(trues_real, baseline_real)

print("\n📌 BASELINE (Forecast only)")
print(f"MAE : {mae_b:.4f}")
print(f"MSE : {mse_b:.4f}")
print(f"RMSE: {rmse_b:.4f}")
print(f"R2  : {r2_b:.4f}")


# =========================
# SAVE FILES FOR PAPER/PLOTS
# =========================
os.makedirs("outputs", exist_ok=True)

pred_df = pd.DataFrame({
    "true_O3": trues_real[:, 0],
    "pred_O3": preds_real[:, 0],
    "true_NO2": trues_real[:, 1],
    "pred_NO2": preds_real[:, 1],
    "forecast_O3": forecast[:, 0],
    "forecast_NO2": forecast[:, 1],
})

pred_df.to_csv("outputs/predictions_test.csv", index=False)

summary_df = pd.DataFrame([{
    "model_mae": mae, "model_mse": mse, "model_rmse": rmse, "model_r2": r2,
    "baseline_mae": mae_b, "baseline_mse": mse_b, "baseline_rmse": rmse_b, "baseline_r2": r2_b,
}])

summary_df.to_csv("outputs/metrics_summary.csv", index=False)

print("\n✅ Saved:")
print(" - outputs/predictions_test.csv")
print(" - outputs/metrics_summary.csv")