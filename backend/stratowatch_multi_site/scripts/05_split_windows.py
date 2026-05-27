"""
05_split_windows.py

Purpose:
--------
Split windowed samples into train/val/test WITHOUT leakage.
Time-series rule: earliest windows for training, latest for testing.

Input:
- data/processed/train_windows_Tin24_Tout6_stride1.npz

Output:
- data/processed/splits_Tin24_Tout6_stride1.npz
  containing:
    X_train, Y_train, masks
    X_val, Y_val, masks
    X_test, Y_test, masks
"""

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_windows_Tin24_Tout6_stride1.npz")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "splits_Tin24_Tout6_stride1.npz")

# Split ratios (time-ordered)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # remaining

data = np.load(IN_PATH, allow_pickle=True)

X = data["X"]
Y = data["Y"]
X_mask = data["X_mask"]
Y_mask = data["Y_mask"]

N = X.shape[0]
print("Total windows:", N)

n_train = int(N * TRAIN_RATIO)
n_val = int(N * VAL_RATIO)
n_test = N - n_train - n_val  # remainder

print("Train:", n_train, "Val:", n_val, "Test:", n_test)

# Time-ordered slicing
X_train, Y_train = X[:n_train], Y[:n_train]
Xm_train, Ym_train = X_mask[:n_train], Y_mask[:n_train]

X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
Xm_val, Ym_val = X_mask[n_train:n_train+n_val], Y_mask[n_train:n_train+n_val]

X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
Xm_test, Ym_test = X_mask[n_train+n_val:], Y_mask[n_train+n_val:]

# Save
np.savez_compressed(
    OUT_PATH,
    X_train=X_train, Y_train=Y_train, X_mask_train=Xm_train, Y_mask_train=Ym_train,
    X_val=X_val, Y_val=Y_val, X_mask_val=Xm_val, Y_mask_val=Ym_val,
    X_test=X_test, Y_test=Y_test, X_mask_test=Xm_test, Y_mask_test=Ym_test,
    sites=data["sites"],
    feature_cols=data["feature_cols"],
    target_cols=data["target_cols"],
    tin=data["tin"], tout=data["tout"], stride=data["stride"],
)

print("✅ Saved splits to:", OUT_PATH)
print("Shapes:")
print("Train X:", X_train.shape, "Train Y:", Y_train.shape)
print("Val   X:", X_val.shape, "Val   Y:", Y_val.shape)
print("Test  X:", X_test.shape, "Test  Y:", Y_test.shape)