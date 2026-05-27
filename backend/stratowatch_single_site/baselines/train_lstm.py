# baselines/train_lstm.py
# Purpose:
#   Train an LSTM baseline on sequence input (N, T, F) -> residual targets (N, 2)
#   Then reconstruct real targets: y = forecast + residual
#
# Run:
#   python3 -m baselines.train_lstm

import os
import json
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from baselines.common_data import load_splits, load_feature_list_and_forecast_from_X


# -------------------------
# LSTM Model
# -------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.2, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)       # out: (B, T, H)
        last = out[:, -1, :]        # last timestep representation: (B, H)
        return self.head(last)      # (B, 2)


def compute_metrics(trues_real, preds_real, prefix=""):
    mae = mean_absolute_error(trues_real, preds_real)
    mse = mean_squared_error(trues_real, preds_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(trues_real, preds_real)

    print(f"\n✅ {prefix} METRICS (Real Units)")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
    return mae, rmse, r2


def main():
    os.makedirs("outputs/baselines", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("✅ Using device:", device)

    # 1) Load splits
    splits = load_splits()
    X_train_seq = splits["X_train_seq"] if "X_train_seq" in splits else None
    X_val_seq   = splits["X_val_seq"] if "X_val_seq" in splits else None
    X_test_seq  = splits["X_test_seq"]  # (N_test, T, F)

    # NOTE: our load_splits currently returns only X_test_seq.
    # So we rebuild train/val seqs from sequences.npz here for simplicity.
    # We'll load full sequences again:
    data = np.load("data/sequences.npz")
    X_seq = data["X_seq"]
    y_seq = data["y_seq"]
    N, T, F = X_seq.shape

    test_size = int(N * 0.15)
    val_size  = int(N * 0.15)
    train_size = N - val_size - test_size

    X_train_seq = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val_seq = X_seq[train_size:train_size + val_size]
    y_val = y_seq[train_size:train_size + val_size]
    X_test_seq = X_seq[train_size + val_size:]
    y_test = y_seq[train_size + val_size:]

    print("✅ Loaded sequences:")
    print("Train:", X_train_seq.shape, y_train.shape)
    print("Val  :", X_val_seq.shape, y_val.shape)
    print("Test :", X_test_seq.shape, y_test.shape)

    # 2) DataLoaders
    BATCH = 128

    train_ds = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32),
                             torch.tensor(y_val, dtype=torch.float32))
    test_ds  = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    # 3) Build model
    model = LSTMRegressor(in_dim=F, hidden_dim=128, num_layers=2, dropout=0.2, out_dim=2).to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_val = float("inf")
    patience = 5
    counter = 0
    ckpt_path = "outputs/checkpoints/best_lstm.pt"

    train_losses, val_losses = [], []

    # 4) Train
    EPOCHS = 30
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_loader.dataset)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), ckpt_path)
            print("✅ Saved best LSTM checkpoint")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("⏹ Early stopping triggered")
                break

    # Save loss curve
    import matplotlib.pyplot as plt
    os.makedirs("outputs/plots", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("LSTM Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/lstm_loss_curve.png", dpi=300)
    plt.close()
    print("✅ Saved: outputs/plots/lstm_loss_curve.png")

    # 5) Predict scaled residuals
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds_scaled.append(pred)
            trues_scaled.append(yb.numpy())

    preds_scaled = np.vstack(preds_scaled)
    trues_scaled = np.vstack(trues_scaled)

    # 6) Convert residuals back to real units
    y_res_scaler = joblib.load("data/y_res_scaler.pkl")
    preds_residual = y_res_scaler.inverse_transform(preds_scaled)
    trues_residual = y_res_scaler.inverse_transform(trues_scaled)

    # 7) Forecast extraction from input last step
    _, forecast = load_feature_list_and_forecast_from_X(
        X_test_seq_original=X_test_seq,
        feature_list_path="data/feature_list.json",
        o3_feature_name="O3_forecast",
        no2_feature_name="NO2_forecast"
    )

    preds_real = forecast + preds_residual
    trues_real = forecast + trues_residual

    # 8) Metrics
    mae, rmse, r2 = compute_metrics(trues_real, preds_real, prefix="LSTM")

    # Per target
    for i, name in enumerate(["O3_target", "NO2_target"]):
        compute_metrics(trues_real[:, i], preds_real[:, i], prefix=f"LSTM {name}")

    # Save outputs
    np.save("outputs/baselines/lstm_preds_real.npy", preds_real)
    np.save("outputs/baselines/lstm_true_real.npy", trues_real)

    with open("outputs/baselines/lstm_paper_row.csv", "w") as f:
        f.write("Method,MAE,RMSE,R2\n")
        f.write(f"LSTM (Residual + Forecast),{mae:.6f},{rmse:.6f},{r2:.6f}\n")

    print("\n✅ Saved LSTM outputs to outputs/baselines/")


if __name__ == "__main__":
    main()