import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SINGLE_ROOT = ROOT / "backend" / "stratowatch_single_site"
MULTI_ROOT = ROOT / "backend" / "stratowatch_multi_site"
sys.path.insert(0, str(SINGLE_ROOT))
sys.path.insert(0, str(MULTI_ROOT))

from models.temporal_transformer import TemporalTransformer  # noqa: E402
from src.models.graph_st_transformer import GraphSTTransformer  # noqa: E402
from src.models.st_transformer import STTransformer  # noqa: E402


class ModelContractTests(unittest.TestCase):
    def test_single_site_model_and_checkpoint(self):
        model = TemporalTransformer(in_dim=179, d_model=128, nhead=8, num_layers=3, out_dim=2)
        checkpoint = SINGLE_ROOT / "outputs" / "checkpoints" / "best_model.pt"
        self.assertTrue(checkpoint.is_file())
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state)

        with torch.no_grad():
            output = model(torch.zeros(1, 24, 179))
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_multi_site_models_and_checkpoints(self):
        st = STTransformer(num_features=28, tin=24, tout=6, num_sites=7)
        graph = GraphSTTransformer(num_features=28, tin=24, tout=6, num_sites=7)

        st_checkpoint = MULTI_ROOT / "outputs" / "checkpoints" / "st_transformer_best.pt"
        graph_checkpoint = MULTI_ROOT / "outputs" / "checkpoints" / "graph_st_best.pt"
        self.assertTrue(st_checkpoint.is_file())
        self.assertTrue(graph_checkpoint.is_file())

        st_payload = torch.load(st_checkpoint, map_location="cpu", weights_only=False)
        graph_payload = torch.load(graph_checkpoint, map_location="cpu", weights_only=False)
        st.load_state_dict(st_payload["model_state"])
        graph.load_state_dict(graph_payload["model_state"])

        features = torch.zeros(1, 24, 7, 28)
        adjacency = torch.from_numpy(__import__("numpy").load(
            MULTI_ROOT / "data" / "processed" / "adjacency_final.npy"
        )).float()
        coordinates = torch.from_numpy(__import__("numpy").load(
            MULTI_ROOT / "configs" / "site_coords_raw.npy"
        )).float()

        with torch.no_grad():
            st_output, _ = st(features)
            graph_output = graph(features, adjacency, coordinates)
        self.assertEqual(tuple(st_output.shape), (1, 6, 7, 2))
        self.assertEqual(tuple(graph_output.shape), (1, 6, 7, 2))

    def test_residual_reconstruction_contract_is_present(self):
        source = (SINGLE_ROOT / "evaluate.py").read_text()
        self.assertIn("preds_real = forecast + preds_residual", source)
        self.assertIn("trues_real = forecast + trues_residual", source)


if __name__ == "__main__":
    unittest.main()