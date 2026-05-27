# models/temporal_transformer.py
# Purpose:
#   Define a baseline Temporal Transformer for your single-site air-quality forecasting.
#   Input:  (B, T, F)  -> B=batch size, T=24 time steps, F=179 features
#   Output: (B, 2)     -> predict next-step targets (O3_target, NO2_target)

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer baseline (single-site).
    Learns temporal dependencies using Transformer Encoder.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        out_dim: int = 2,
    ):
        super().__init__()

        # 1) Vector embedding (feature vector -> transformer embedding dimension)
        # Converts 179 features into d_model (ex: 128) dimensional embedding
        self.input_proj = nn.Linear(in_dim, d_model)

        # 2) Transformer Encoder (temporal attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) Output head (predict next-step targets from last time step representation)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns: (B, out_dim)
        """
        x = self.input_proj(x)   # (B, T, d_model)
        h = self.encoder(x)      # (B, T, d_model)

        # Use last time-step embedding to predict next step
        last = h[:, -1, :]       # (B, d_model)
        yhat = self.head(last)   # (B, out_dim)
        return yhat