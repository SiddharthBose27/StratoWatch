"""
03_align_multisite.py

Purpose:
--------
Create one common hourly timeline for all 7 sites, and build multi-site arrays:

Train:
- X       : (T, S, F)  inputs (ALL non-target columns kept)
- Y       : (T, S, 2)  targets (O3_target, NO2_target)
- X_mask  : (T, S, F)  1 = present, 0 = missing
- Y_mask  : (T, S, 2)

Unseen:
- X       : (T, S, F)
- X_mask  : (T, S, F)

Notes:
------
- We DO NOT remove any features.
- We use UNION of timestamps across sites.
- We fill missing values with 0.0 but keep masks to tell the model what is missing.
"""

import os
import re
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "data")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COLS = ["O3_target", "NO2_target"]

def is_real_csv(filename: str) -> bool:
    return filename.endswith(".csv") and not filename.startswith("._")

def extract_site_id(fname: str) -> int:
    m = re.search(r"site_(\d+)_", fname)
    return int(m.group(1)) if m else -1

def load_site(path: str) -> pd.DataFrame:
    """
    Load one site CSV and create a datetime index.
    We keep year/month/day/hour columns as FEATURES (do not drop them).
    """
    df = pd.read_csv(path)

    dt = pd.to_datetime(
        dict(
            year=df["year"].astype(int),
            month=df["month"].astype(int),
            day=df["day"].astype(int),
            hour=df["hour"].astype(int),
        )
    )
    df.insert(0, "datetime", dt)
    df = df.set_index("datetime").sort_index()
    return df

def union_timeline(dfs: dict[int, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = None
    for df in dfs.values():
        idx = df.index if idx is None else idx.union(df.index)
    return idx.sort_values()

# -------- Collect files --------
all_files = [f for f in os.listdir(RAW_DIR) if is_real_csv(f)]
train_files = sorted([f for f in all_files if f.endswith("_train_data.csv")])
unseen_files = sorted([f for f in all_files if f.endswith("_unseen_input_data.csv")])

sites = sorted([extract_site_id(f) for f in train_files])

print("Sites:", sites)

# -------- Load per-site dataframes --------
train_dfs = {}
unseen_dfs = {}

for f in train_files:
    sid = extract_site_id(f)
    train_dfs[sid] = load_site(os.path.join(RAW_DIR, f))

for f in unseen_files:
    sid = extract_site_id(f)
    unseen_dfs[sid] = load_site(os.path.join(RAW_DIR, f))

# -------- Decide feature columns --------
# We keep ALL columns except the targets.
all_columns = list(train_dfs[sites[0]].columns)
FEATURE_COLS = [c for c in all_columns if c not in TARGET_COLS]

print("Num features:", len(FEATURE_COLS))
print("Features:", FEATURE_COLS)

# -------- Build UNION timelines --------
train_time = union_timeline(train_dfs)
unseen_time = union_timeline(unseen_dfs)

print("Train timeline:", train_time.min(), "->", train_time.max(), "T =", len(train_time))
print("Unseen timeline:", unseen_time.min(), "->", unseen_time.max(), "T =", len(unseen_time))

# -------- Helper to build arrays --------
def build_arrays(dfs: dict[int, pd.DataFrame], timeline: pd.DatetimeIndex, is_train: bool):
    T = len(timeline)
    S = len(sites)
    F = len(FEATURE_COLS)

    X = np.zeros((T, S, F), dtype=np.float32)
    X_mask = np.zeros((T, S, F), dtype=np.float32)

    if is_train:
        Y = np.zeros((T, S, len(TARGET_COLS)), dtype=np.float32)
        Y_mask = np.zeros((T, S, len(TARGET_COLS)), dtype=np.float32)
    else:
        Y, Y_mask = None, None

    for s_idx, sid in enumerate(sites):
        df = dfs[sid].reindex(timeline)

        # X
        x_vals = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        x_mask = (~np.isnan(x_vals)).astype(np.float32)
        x_vals = np.nan_to_num(x_vals, nan=0.0)

        X[:, s_idx, :] = x_vals
        X_mask[:, s_idx, :] = x_mask

        if is_train:
            y_vals = df[TARGET_COLS].to_numpy(dtype=np.float32)
            y_mask = (~np.isnan(y_vals)).astype(np.float32)
            y_vals = np.nan_to_num(y_vals, nan=0.0)

            Y[:, s_idx, :] = y_vals
            Y_mask[:, s_idx, :] = y_mask

    return X, X_mask, Y, Y_mask

# -------- Build and save --------
X_train, X_train_mask, Y_train, Y_train_mask = build_arrays(train_dfs, train_time, is_train=True)
X_unseen, X_unseen_mask, _, _ = build_arrays(unseen_dfs, unseen_time, is_train=False)

# Save timestamps as int64 (ns) for easy restore
train_time_int = train_time.view("int64")
unseen_time_int = unseen_time.view("int64")

np.savez_compressed(
    os.path.join(OUT_DIR, "train_aligned.npz"),
    X=X_train,
    X_mask=X_train_mask,
    Y=Y_train,
    Y_mask=Y_train_mask,
    sites=np.array(sites, dtype=np.int32),
    feature_cols=np.array(FEATURE_COLS, dtype=object),
    target_cols=np.array(TARGET_COLS, dtype=object),
    time=train_time_int,
)

np.savez_compressed(
    os.path.join(OUT_DIR, "unseen_aligned.npz"),
    X=X_unseen,
    X_mask=X_unseen_mask,
    sites=np.array(sites, dtype=np.int32),
    feature_cols=np.array(FEATURE_COLS, dtype=object),
    time=unseen_time_int,
)

# Save small metadata json
meta = {
    "sites": sites,
    "feature_cols": FEATURE_COLS,
    "target_cols": TARGET_COLS,
    "train_T": int(X_train.shape[0]),
    "unseen_T": int(X_unseen.shape[0]),
}
with open(os.path.join(OUT_DIR, "aligned_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Saved:")
print(" - data/processed/train_aligned.npz")
print(" - data/processed/unseen_aligned.npz")
print(" - data/processed/aligned_meta.json")
print("Train X shape:", X_train.shape, "Train Y shape:", Y_train.shape)
print("Unseen X shape:", X_unseen.shape)