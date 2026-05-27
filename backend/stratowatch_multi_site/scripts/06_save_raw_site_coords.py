"""
scripts/06_save_raw_site_coords.py

Saves RAW (unscaled) site coordinates to:
configs/site_coords_raw.npy

Why:
- Your dataset's site_lat/site_lon were standardized (scaled)
- For geometry (direction vectors), we must use raw lat/lon degrees
"""

import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

coords = np.array([
    [28.69536, 77.18168],  # site 1
    [28.57180, 77.07125],  # site 2
    [28.58278, 77.23441],  # site 3
    [28.82286, 77.10197],  # site 4
    [28.53077, 77.27123],  # site 5
    [28.72954, 77.09601],  # site 6
    [28.71052, 77.24951],  # site 7
], dtype=np.float32)

out_path = os.path.join(PROJECT_ROOT, "configs", "site_coords_raw.npy")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

np.save(out_path, coords)

print("✅ Saved:", out_path)
print("Shape:", coords.shape)
print(coords)