# =========================
# train.py
# Purpose:
#   Train a Transformer on (X_seq -> y_seq) created during preprocessing.
# =========================

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import your model from models folder
from models.temporal_transformer import TemporalTransformer


# =========================
# STEP 0: CONFIG (Change these values for experiments)
# =========================
DATA_PATH = "data/sequences.npz"

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4


VAL_RATIO = 0.15
TEST_RATIO = 0.15

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3



# =========================
# STEP 1: DEVICE SELECTION (Mac M1)
# =========================
# On Mac M1, PyTorch can use Apple's GPU using "mps"
# If mps is unavailable, fallback to CPU.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("✅ Using device:", device)


# =========================
# STEP 2: LOAD DATA (X_seq, y_seq)
# =========================
data = np.load(DATA_PATH)

X_seq = data["X_seq"]   # shape: (N, T, F)
y_seq = data["y_seq"]   # shape: (N, 2)

N, T, F = X_seq.shape
print("✅ Loaded data:", X_seq.shape, y_seq.shape)

# =========================
# SANITY CHECK: detect NaN/Inf and extreme values
# =========================
print("X finite:", np.isfinite(X_seq).all(), " | y finite:", np.isfinite(y_seq).all())
print("NaNs in X:", np.isnan(X_seq).sum(), " | NaNs in y:", np.isnan(y_seq).sum())
print("Infs in X:", np.isinf(X_seq).sum(), " | Infs in y:", np.isinf(y_seq).sum())
print("X max abs:", np.nanmax(np.abs(X_seq)), " | y max abs:", np.nanmax(np.abs(y_seq)))


# =========================
# STEP 3: TIME-BASED SPLIT (No random split to avoid leakage)
# =========================
test_size = int(N * TEST_RATIO)
val_size  = int(N * VAL_RATIO)
train_size = N - val_size - test_size

X_train, y_train = X_seq[:train_size], y_seq[:train_size]
X_val,   y_val   = X_seq[train_size:train_size + val_size], y_seq[train_size:train_size + val_size]
X_test,  y_test  = X_seq[train_size + val_size:], y_seq[train_size + val_size:]

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# =========================
# STEP 4: CREATE PYTORCH DATALOADERS
# =========================
# Convert numpy arrays to torch tensors
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Batches | Train:", len(train_loader), "Val:", len(val_loader), "Test:", len(test_loader))


# =========================
# STEP 5: BUILD MODEL
# =========================
# Your model converts F=179 -> d_model embeddings and applies Transformer encoder.
# Predicts 2 values (O3_target, NO2_target).
model = TemporalTransformer(
    in_dim=F,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    out_dim=2
).to(device)

# Loss + Optimizer
criterion = torch.nn.MSELoss()  # regression
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# If val loss stops improving, reduce LR automatically
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,     # LR becomes half
    patience=2,     # wait 2 epochs with no improvement
)

# Save best model checkpoint
os.makedirs("outputs/checkpoints", exist_ok=True)
# Store losses for plotting
train_losses = []
val_losses = []
best_val_loss = float("inf")

patience = 5
counter = 0

# =========================
# STEP 6: TRAINING LOOP
# =========================
for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)           # forward pass
        loss = criterion(pred, yb) # compute loss
        loss.backward()            # backprop
        optimizer.step()           # update weights

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    # Update LR based on validation loss
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "outputs/checkpoints/best_model.pt")
        print("✅ Saved best checkpoint")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("⏹ Early stopping triggered")
            break

# =========================
# STEP 6.5: Save Loss Curve Plot
# =========================
import matplotlib.pyplot as plt

os.makedirs("outputs/plots", exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss (MSE)")
plt.plot(val_losses, label="Val Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/loss_curve.png", dpi=300)
plt.close()

print("✅ Saved loss curve -> outputs/plots/loss_curve.png")


# =========================
# STEP 7: TEST EVALUATION (Residual learning -> Real Targets)
# =========================
import joblib
import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ckpt_path = "outputs/checkpoints/best_model.pt"
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print("⚠️ No checkpoint found. Using last epoch weights.")

model.eval()

preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()   # scaled residual predictions
        preds.append(pred)
        trues.append(yb.numpy())         # scaled residual ground truth

preds = np.vstack(preds)   # (N_test, 2)
trues = np.vstack(trues)   # (N_test, 2)

# -------------------------
# 1) Convert residuals back to real units
# -------------------------
y_res_scaler = joblib.load("data/y_res_scaler.pkl")
preds_residual = y_res_scaler.inverse_transform(preds)  # (N_test, 2) residual in real units
trues_residual = y_res_scaler.inverse_transform(trues)

# -------------------------
# 2) Get forecast values from input (last time step)
# -------------------------
with open("data/feature_list.json", "r") as f:
    feature_list = json.load(f)

o3_idx  = feature_list.index("O3_forecast")
no2_idx = feature_list.index("NO2_forecast")

# test_ds.tensors[0] shape: (N_test, T, F)
X_test_tensor = test_ds.tensors[0].numpy()

forecast_o3  = X_test_tensor[:, -1, o3_idx]
forecast_no2 = X_test_tensor[:, -1, no2_idx]
forecast = np.stack([forecast_o3, forecast_no2], axis=1)  # (N_test, 2)

# -------------------------
# 3) Reconstruct final targets in real units
# y = forecast + residual
# -------------------------
preds_real = forecast + preds_residual
trues_real = forecast + trues_residual

# -------------------------
# 4) Compute metrics on REAL targets
# -------------------------
mae = mean_absolute_error(trues_real, preds_real)
mse = mean_squared_error(trues_real, preds_real)
rmse = np.sqrt(mse)
r2 = r2_score(trues_real, preds_real)

print("\n✅ FINAL TEST METRICS (Real Units, Residual Learning)")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# -------------------------
# 5) Per-target metrics
# -------------------------
target_names = ["O3_target", "NO2_target"]
for i, name in enumerate(target_names):
    mae_i = mean_absolute_error(trues_real[:, i], preds_real[:, i])
    rmse_i = np.sqrt(mean_squared_error(trues_real[:, i], preds_real[:, i]))
    r2_i = r2_score(trues_real[:, i], preds_real[:, i])

    print(f"\n📌 Metrics for {name} (Residual Learning)")
    print(f"MAE : {mae_i:.4f}")
    print(f"RMSE: {rmse_i:.4f}")
    print(f"R2  : {r2_i:.4f}")

# -------------------------
# STEP 2 (Baseline): Forecast-only baseline (delta = 0)
# -------------------------
baseline_real = forecast  # if model predicts residual=0, prediction is just forecast

mae_b = mean_absolute_error(trues_real, baseline_real)
rmse_b = np.sqrt(mean_squared_error(trues_real, baseline_real))
r2_b = r2_score(trues_real, baseline_real)

print("\n📌 FORECAST-ONLY BASELINE (Real Units)")
print(f"MAE : {mae_b:.4f}")
print(f"RMSE: {rmse_b:.4f}")
print(f"R2  : {r2_b:.4f}")