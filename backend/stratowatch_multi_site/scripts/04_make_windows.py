"""
04_make_windows.py

Purpose:
--------
Convert aligned multi-site arrays into sliding windows:

Input:
- data/processed/train_aligned.npz

Output:
- data/processed/train_windows_Tin24_Tout6_stride1.npz

We keep:
- X, Y
- masks (X_mask, Y_mask)

Shapes:
- X_win: (N, Tin, S, F)
- Y_win: (N, Tout, S, 2)
"""

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_aligned.npz")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Window params (optimal for M1 8GB) ----
TIN = 24
TOUT = 6
STRIDE = 1

data = np.load(IN_PATH, allow_pickle=True)

X = data["X"]          # (T, S, F)
X_mask = data["X_mask"]
Y = data["Y"]          # (T, S, 2)
Y_mask = data["Y_mask"]

sites = data["sites"]
feature_cols = data["feature_cols"]
target_cols = data["target_cols"]
time = data["time"]

T, S, F = X.shape
print("Loaded X:", X.shape, "Y:", Y.shape)

# Number of windows
N = (T - (TIN + TOUT) + 1) // STRIDE
print("Total windows:", N)

# Allocate
X_win = np.zeros((N, TIN, S, F), dtype=np.float32)
Y_win = np.zeros((N, TOUT, S, Y.shape[-1]), dtype=np.float32)

X_mask_win = np.zeros_like(X_win, dtype=np.float32)
Y_mask_win = np.zeros_like(Y_win, dtype=np.float32)

# Build windows
for i in range(N):
    t0 = i * STRIDE
    X_win[i] = X[t0 : t0 + TIN]
    X_mask_win[i] = X_mask[t0 : t0 + TIN]

    Y_win[i] = Y[t0 + TIN : t0 + TIN + TOUT]
    Y_mask_win[i] = Y_mask[t0 + TIN : t0 + TIN + TOUT]

# Save
out_path = os.path.join(OUT_DIR, f"train_windows_Tin{TIN}_Tout{TOUT}_stride{STRIDE}.npz")
np.savez_compressed(
    out_path,
    X=X_win, Y=Y_win,
    X_mask=X_mask_win, Y_mask=Y_mask_win,
    sites=sites,
    feature_cols=feature_cols,
    target_cols=target_cols,
    tin=np.array([TIN], dtype=np.int32),
    tout=np.array([TOUT], dtype=np.int32),
    stride=np.array([STRIDE], dtype=np.int32),
)

print("✅ Saved windows to:", out_path)
print("X_win shape:", X_win.shape)
print("Y_win shape:", Y_win.shape)