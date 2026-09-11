"""
Final StratoWatch single-site Temporal Transformer.

Contract:
    Input:  (B, Tin=24, F=134)
    Output: (B, Tout=6, targets=2)

The model predicts residuals relative to the explicit forecast baseline.
Final target reconstruction is:

    target_hat = forecast_baseline + predicted_residual
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        out_dim: int = 2,
        tin: int = 24,
        tout: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.d_model = d_model
        self.tin = tin
        self.tout = tout
        self.out_dim = out_dim

        self.input_projection = nn.Linear(in_dim, d_model)

        self.time_pos = nn.Parameter(
            torch.zeros(1, tin, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, tout * out_dim),
        )

        nn.init.normal_(self.time_pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (B, Tin, F)

        returns:
            residual prediction:
            (B, Tout, out_dim)
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, Tin, F), got {tuple(x.shape)}"
            )

        B, Tin, F = x.shape

        if Tin != self.tin:
            raise ValueError(
                f"Expected Tin={self.tin}, got {Tin}"
            )

        if F != self.in_dim:
            raise ValueError(
                f"Expected F={self.in_dim}, got {F}"
            )

        h = self.input_projection(x)
        h = h + self.time_pos[:, :Tin, :]

        h = self.encoder(h)

        # Representation of the latest observed hour.
        h_last = h[:, -1, :]

        out = self.head(h_last)

        out = out.view(
            B,
            self.tout,
            self.out_dim,
        )

        return out