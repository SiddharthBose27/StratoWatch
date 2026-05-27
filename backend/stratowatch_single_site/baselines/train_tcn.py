# baselines/train_tcn.py
# Purpose:
#   Train a TCN (Temporal Convolutional Network) baseline on sequence input (N, T, F)
#   Predict residual targets (N, 2) then reconstruct real targets: y = forecast + residual
#
# Run:
#   python3 -m baselines.train_tcn

import os
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from baselines.common_data import load_feature_list_and_forecast_from_X


# -------------------------
# TCN Building Blocks
# -------------------------
class Chomp1d(nn.Module):
    """Remove extra padding to keep causal convolution output length = input length."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (B, C, T)
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, in_dim, channels=(64, 64, 64), kernel_size=3, dropout=0.2, out_dim=2):
        super().__init__()
        layers = []
        in_ch = in_dim
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], out_dim),
        )

    def forward(self, x):
        # x: (B, T, F) -> Conv1d expects (B, C, T)
        x = x.transpose(1, 2)  # (B, F, T)
        h = self.tcn(x)        # (B, C, T)
        last = h[:, :, -1]     # last timestep: (B, C)
        return self.head(last) # (B, 2)


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
    os.makedirs("outputs/plots", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("✅ Using device:", device)

    # Load full sequences (same split logic)
    data = np.load("data/sequences.npz")
    X_seq = data["X_seq"]
    y_seq = data["y_seq"]
    N, T, F = X_seq.shape

    test_size = int(N * 0.15)
    val_size  = int(N * 0.15)
    train_size = N - val_size - test_size

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size:train_size + val_size]
    y_val = y_seq[train_size:train_size + val_size]
    X_test = X_seq[train_size + val_size:]
    y_test = y_seq[train_size + val_size:]

    print("✅ Loaded sequences:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    # Dataloaders
    BATCH = 128
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.float32)),
                             batch_size=BATCH, shuffle=False)

    # Model
    model = TCN(in_dim=F, channels=(64, 64, 64), kernel_size=3, dropout=0.2, out_dim=2).to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_val = float("inf")
    patience = 5
    counter = 0
    ckpt_path = "outputs/checkpoints/best_tcn.pt"

    train_losses, val_losses = [], []

    # Train
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
            print("✅ Saved best TCN checkpoint")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("⏹ Early stopping triggered")
                break

    # Save loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("TCN Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/tcn_loss_curve.png", dpi=300)
    plt.close()
    print("✅ Saved: outputs/plots/tcn_loss_curve.png")

    # Predict
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

    # Convert residuals back to real units
    y_res_scaler = joblib.load("data/y_res_scaler.pkl")
    preds_residual = y_res_scaler.inverse_transform(preds_scaled)
    trues_residual = y_res_scaler.inverse_transform(trues_scaled)

    # Forecast extraction (last timestep)
    _, forecast = load_feature_list_and_forecast_from_X(
        X_test_seq_original=X_test,
        feature_list_path="data/feature_list.json",
        o3_feature_name="O3_forecast",
        no2_feature_name="NO2_forecast"
    )

    preds_real = forecast + preds_residual
    trues_real = forecast + trues_residual

    # Metrics
    mae, rmse, r2 = compute_metrics(trues_real, preds_real, prefix="TCN")

    for i, name in enumerate(["O3_target", "NO2_target"]):
        compute_metrics(trues_real[:, i], preds_real[:, i], prefix=f"TCN {name}")

    # Save
    np.save("outputs/baselines/tcn_preds_real.npy", preds_real)
    np.save("outputs/baselines/tcn_true_real.npy", trues_real)

    with open("outputs/baselines/tcn_paper_row.csv", "w") as f:
        f.write("Method,MAE,RMSE,R2\n")
        f.write(f"TCN (Residual + Forecast),{mae:.6f},{rmse:.6f},{r2:.6f}\n")

    print("\n✅ Saved TCN outputs to outputs/baselines/")


if __name__ == "__main__":
    main()