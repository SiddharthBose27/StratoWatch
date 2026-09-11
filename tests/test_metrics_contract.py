import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "stratowatch_multi_site"))

from src.utils.metrics import masked_huber, masked_mae, masked_mse, masked_rmse  # noqa: E402


class MetricsContractTests(unittest.TestCase):
    def setUp(self):
        self.prediction = torch.tensor([[[[2.0], [4.0]]]])
        self.target = torch.tensor([[[[1.0], [1.0]]]])
        self.mask = torch.tensor([[[[1.0], [0.0]]]])

    def test_masked_mae_mse_rmse(self):
        self.assertAlmostEqual(masked_mae(self.prediction, self.target, self.mask).item(), 1.0)
        self.assertAlmostEqual(masked_mse(self.prediction, self.target, self.mask).item(), 1.0)
        self.assertAlmostEqual(masked_rmse(self.prediction, self.target, self.mask).item(), 1.0)

    def test_masked_huber_uses_delta_one(self):
        self.assertAlmostEqual(masked_huber(self.prediction, self.target, self.mask).item(), 0.5)

    def test_empty_mask_returns_zero(self):
        empty_mask = torch.zeros_like(self.mask)
        self.assertEqual(masked_mae(self.prediction, self.target, empty_mask).item(), 0.0)
        self.assertEqual(masked_rmse(self.prediction, self.target, empty_mask).item(), 0.0)

    def test_evaluators_average_batch_scalars(self):
        for relative in [
            "src/eval_baseline_realunits.py",
            "src/eval_graph_realunits.py",
        ]:
            source = (ROOT / "backend" / "stratowatch_multi_site" / relative).read_text()
            self.assertIn("mae_sum +=", source)
            self.assertIn("rmse_sum +=", source)
            self.assertIn("mae_sum / n", source)
            self.assertIn("rmse_sum / n", source)


if __name__ == "__main__":
    unittest.main()