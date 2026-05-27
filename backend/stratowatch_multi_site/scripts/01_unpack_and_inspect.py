"""
01_unpack_and_inspect.py

Purpose:
--------
1. Unzip the dataset from data/raw/
2. Print basic information about the files
3. Check one site CSV structure

We are NOT modifying anything yet.
Just inspecting.
"""

import os
import zipfile
import pandas as pd

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "data", "raw")

print("Base directory:", BASE_DIR)
print("Zip file path:", RAW_DATA_PATH)


# -------- Step 1: Unzip --------
if os.path.exists(RAW_DATA_PATH):
    print("Unzipping dataset...")
    with zipfile.ZipFile(RAW_DATA_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("Unzip complete.")
else:
    print("data.zip not found. Please check path.")
    exit()


# -------- Step 2: List extracted files --------
print("\nListing extracted files:")
for root, dirs, files in os.walk(EXTRACT_PATH):
    for file in files:
        if file.endswith(".csv"):
            print(file)


# -------- Step 3: Inspect one file --------
sample_file = os.path.join(EXTRACT_PATH, "data", "site_1_train_data.csv")

if os.path.exists(sample_file):
    df = pd.read_csv(sample_file)
    print("\nSample file shape:", df.shape)
    print("\nColumns:")
    print(df.columns)
    print("\nFirst 5 rows:")
    print(df.head())
else:
    print("Sample file not found.")