import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "backend" / "stratowatch_multi_site" / "data" / "raw" / "data"
MULTI = ROOT / "backend" / "stratowatch_multi_site"


class Phase2PipelineSmokeTests(unittest.TestCase):
    def test_raw_site_inventory_and_schema(self):
        train_files = sorted(RAW.glob("site_*_train_data.csv"))
        unseen_files = sorted(RAW.glob("site_*_unseen_input_data.csv"))
        self.assertEqual(len(train_files), 7)
        self.assertEqual(len(unseen_files), 7)
        self.assertEqual([p.name.split("_")[1] for p in train_files], [str(i) for i in range(1, 8)])

        expected_train = [
            "year", "month", "day", "hour", "O3_forecast", "NO2_forecast",
            "T_forecast", "q_forecast", "u_forecast", "v_forecast", "w_forecast",
            "NO2_satellite", "HCHO_satellite", "ratio_satellite", "O3_target", "NO2_target",
        ]
        expected_unseen = expected_train[:-2]

        for path in train_files:
            frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns), expected_train)
            timestamps = pd.to_datetime(
                dict(year=frame.year, month=frame.month, day=frame.day, hour=frame.hour)
            )
            self.assertEqual(int(timestamps.duplicated().sum()), 0)

        for path in unseen_files:
            frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns), expected_unseen)

    def test_final_artifact_and_scaler_metadata(self):
        npz_path = MULTI / "data" / "processed" / "splits_final_Yscaled_Tin24_Tout6_stride1.npz"
        feature_scaler = json.loads((MULTI / "configs" / "final_feature_scaler.json").read_text())
        global_scaler = json.loads((MULTI / "configs" / "global_scaler.json").read_text())
        target_scaler = json.loads((MULTI / "configs" / "target_scaler.json").read_text())

        with np.load(npz_path, allow_pickle=True) as data:
            self.assertEqual(data["X_train"].shape[-1], 28)
            self.assertEqual(data["Y_train"].shape[-1], 2)
            self.assertEqual(int(data["tin"].reshape(-1)[0]), 24)
            self.assertEqual(int(data["tout"].reshape(-1)[0]), 6)
            self.assertEqual(data["feature_cols"].tolist(), feature_scaler["feature_cols"])
            self.assertEqual(data["target_cols"].tolist(), target_scaler["targets"])

        self.assertEqual(len(global_scaler["feature_cols"]), 14)
        self.assertEqual(len(feature_scaler["feature_cols"]), 28)
        self.assertEqual(feature_scaler["scaled_features"], [
            "NO2_satellite_age", "HCHO_satellite_age", "ratio_satellite_age",
            "site_lat", "site_lon",
        ])

    def test_pipeline_scripts_preserve_declared_parameters(self):
        windows = (MULTI / "scripts" / "04_make_windows.py").read_text()
        splits = (MULTI / "scripts" / "05_split_windows.py").read_text()
        alignment = (MULTI / "scripts" / "03_align_multisite.py").read_text()
        self.assertIn("TIN = 24", windows)
        self.assertIn("TOUT = 6", windows)
        self.assertIn("STRIDE = 1", windows)
        self.assertIn("TRAIN_RATIO = 0.70", splits)
        self.assertIn("VAL_RATIO = 0.15", splits)
        self.assertIn("idx.union(df.index)", alignment)
        self.assertIn("np.nan_to_num(x_vals, nan=0.0)", alignment)
        self.assertIn("np.nan_to_num(y_vals, nan=0.0)", alignment)


if __name__ == "__main__":
    unittest.main()