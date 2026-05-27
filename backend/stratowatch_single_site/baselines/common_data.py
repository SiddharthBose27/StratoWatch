# baselines/common_data.py
# Purpose:
#   Load Transformer-ready sequences and create consistent train/val/test splits
#   for both:
#     - ML models (RF/XGBoost) using flattened windows
#     - Deep models (LSTM/TCN) using 3D sequences
# baselines/common_data.py
# Shared loader for baseline models (RF / XGB / etc.)
# Loads sequences.npz and returns time-based splits.
# Also returns non-flattened X_test_seq for forecast reconstruction.

import os
import json
import numpy as np


def load_splits(data_path="data/sequences.npz", val_ratio=0.15, test_ratio=0.15):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run preprocessing to create it.")

    data = np.load(data_path)
    X_seq = data["X_seq"]  # (N, T, F)
    y_seq = data["y_seq"]  # (N, 2)

    N, T, F = X_seq.shape

    test_size = int(N * test_ratio)
    val_size  = int(N * val_ratio)
    train_size = N - val_size - test_size

    X_train_seq = X_seq[:train_size]
    y_train = y_seq[:train_size]

    X_val_seq = X_seq[train_size:train_size + val_size]
    y_val = y_seq[train_size:train_size + val_size]

    X_test_seq = X_seq[train_size + val_size:]
    y_test = y_seq[train_size + val_size:]

    # Flatten for classical ML baselines
    X_train = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_val   = X_val_seq.reshape(X_val_seq.shape[0], -1)
    X_test  = X_test_seq.reshape(X_test_seq.shape[0], -1)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_test_seq": X_test_seq,  # needed to reconstruct forecast + residual
    }


def load_feature_list_and_forecast_from_X(
    X_test_seq_original,
    feature_list_path="data/feature_list.json",
    o3_feature_name="O3_forecast",
    no2_feature_name="NO2_forecast"
):
    if not os.path.exists(feature_list_path):
        raise FileNotFoundError(f"Missing {feature_list_path}. Run preprocessing to create it.")

    with open(feature_list_path, "r") as f:
        feature_list = json.load(f)

    o3_idx = feature_list.index(o3_feature_name)
    no2_idx = feature_list.index(no2_feature_name)

    # last timestep forecast
    forecast_o3 = X_test_seq_original[:, -1, o3_idx]
    forecast_no2 = X_test_seq_original[:, -1, no2_idx]
    forecast = np.stack([forecast_o3, forecast_no2], axis=1)

    return feature_list, forecast