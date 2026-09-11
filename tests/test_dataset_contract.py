import json
import pickle
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SINGLE_DATA = ROOT / "backend" / "stratowatch_single_site" / "data"
MULTI_DATA = ROOT / "backend" / "stratowatch_multi_site" / "data"


class DatasetContractTests(unittest.TestCase):
    def test_single_site_sequence_schema(self):
        path = SINGLE_DATA / "sequences.npz"
        self.assertTrue(path.is_file())

        with np.load(path) as data:
            self.assertEqual(set(data.files), {"X_seq", "y_seq"})
            self.assertEqual(data["X_seq"].ndim, 3)
            self.assertEqual(data["X_seq"].shape[1:], (24, 179))
            self.assertEqual(data["y_seq"].shape[1:], (2,))

        feature_list = json.loads((SINGLE_DATA / "feature_list.json").read_text())
        self.assertEqual(len(feature_list), 134)
        self.assertLess(feature_list.index("O3_forecast"), len(feature_list))
        self.assertLess(feature_list.index("NO2_forecast"), len(feature_list))

    def test_single_site_feature_list_matches_sequence_width(self):
        with np.load(SINGLE_DATA / "sequences.npz") as data:
            sequence_width = data["X_seq"].shape[-1]
        feature_list = json.loads((SINGLE_DATA / "feature_list.json").read_text())
        self.assertEqual(len(feature_list), sequence_width)

    def test_single_site_residual_scaler_loads(self):
        path = SINGLE_DATA / "y_res_scaler.pkl"
        self.assertTrue(path.is_file())
        with path.open("rb") as handle:
            scaler = pickle.load(handle)
        self.assertEqual(getattr(scaler, "n_features_in_", 2), 2)

    def test_multi_site_final_schema(self):
        path = MULTI_DATA / "processed" / "splits_final_Yscaled_Tin24_Tout6_stride1.npz"
        self.assertTrue(path.is_file())

        with np.load(path, allow_pickle=True) as data:
            expected = {
                "X_train", "X_mask_train", "Y_train", "Y_mask_train",
                "X_val", "X_mask_val", "Y_val", "Y_mask_val",
                "X_test", "X_mask_test", "Y_test", "Y_mask_test",
                "sites", "feature_cols", "target_cols", "tin", "tout", "stride",
            }
            self.assertEqual(set(data.files), expected)
            self.assertEqual(data["X_train"].shape[1:], (24, 7, 28))
            self.assertEqual(data["X_val"].shape[1:], (24, 7, 28))
            self.assertEqual(data["X_test"].shape[1:], (24, 7, 28))
            self.assertEqual(data["Y_train"].shape[1:], (6, 7, 2))
            self.assertEqual(data["Y_val"].shape[1:], (6, 7, 2))
            self.assertEqual(data["Y_test"].shape[1:], (6, 7, 2))
            self.assertEqual(data["X_mask_test"].shape, data["X_test"].shape)
            self.assertEqual(data["Y_mask_test"].shape, data["Y_test"].shape)
            self.assertEqual(data["sites"].tolist(), list(range(1, 8)))
            self.assertEqual(data["target_cols"].tolist(), ["O3_target", "NO2_target"])
            self.assertEqual(int(data["tin"].reshape(-1)[0]), 24)
            self.assertEqual(int(data["tout"].reshape(-1)[0]), 6)


if __name__ == "__main__":
    unittest.main()