"""
02_check_all_sites.py

Purpose:
--------
1) Ignore Mac "._" junk files
2) Verify all 7 sites exist (train + unseen)
3) Confirm columns match across sites
4) Print missing-value percentage for each column (per site)
"""

import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "data")  # after unzip, files are inside data/raw/data/

def is_real_csv(filename: str) -> bool:
    """Return True only for real CSV files (ignore Mac '._' files)."""
    return filename.endswith(".csv") and not filename.startswith("._")

# Collect train & unseen files
all_files = [f for f in os.listdir(DATA_DIR) if is_real_csv(f)]

train_files = sorted([f for f in all_files if f.endswith("_train_data.csv")])
unseen_files = sorted([f for f in all_files if f.endswith("_unseen_input_data.csv")])

print("Train files found:", len(train_files))
print("Unseen files found:", len(unseen_files))
print("Train files:", train_files)
print("Unseen files:", unseen_files)

# Check site ids
def extract_site_id(fname: str) -> int:
    m = re.search(r"site_(\d+)_", fname)
    return int(m.group(1)) if m else -1

train_sites = sorted([extract_site_id(f) for f in train_files])
unseen_sites = sorted([extract_site_id(f) for f in unseen_files])

print("\nTrain site IDs:", train_sites)
print("Unseen site IDs:", unseen_sites)

# Load and compare columns across all sites
train_columns_reference = None

print("\n--- Column consistency check (train) ---")
for f in train_files:
    path = os.path.join(DATA_DIR, f)
    df = pd.read_csv(path)
    cols = list(df.columns)

    if train_columns_reference is None:
        train_columns_reference = cols
        print(f"Reference columns set from: {f}")
    else:
        if cols != train_columns_reference:
            print(f"❌ Column mismatch in {f}")
            print("Expected:", train_columns_reference)
            print("Got     :", cols)
            break

print("✅ Train columns consistent (if no mismatch printed).")

print("\n--- Missing value % per site (train) ---")
for f in train_files:
    path = os.path.join(DATA_DIR, f)
    df = pd.read_csv(path)
    miss_pct = (df.isna().mean() * 100).round(2)
    site = extract_site_id(f)
    print(f"\nSite {site} missing %:")
    print(miss_pct.sort_values(ascending=False))