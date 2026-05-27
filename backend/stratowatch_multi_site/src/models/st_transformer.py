"""
src/models/st_transformer.py

Baseline Spatio-Temporal Transformer (no explicit graph yet).

Input:
  X: (B, Tin, S, F)
  X_mask: (B, Tin, S, F)  [optional for future; for now we don't use it in attention masking]
Output:
  Y_hat: (B, Tout, S, 2)

Design:
- Project features F -> d_model
- Temporal transformer (per site): attends over Tin
- Spatial transformer (per time): attends over S sites
- Forecast head produces Tout steps (direct multi-horizon)

Note:
- This is a strong baseline for CV.
- We'll add graph propagation later (Graph-ST model).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class STTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_targets: int = 2,
        tin: int = 24,
        tout: int = 6,
        num_sites: int = 7,
        d_model: int = 128,
        nhead: int = 4,
        num_layers_time: int = 2,
        num_layers_space: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_targets = num_targets
        self.tin = tin
        self.tout = tout
        self.num_sites = num_sites
        self.d_model = d_model

        # 1) Project raw features into transformer dimension
        self.in_proj = nn.Linear(num_features, d_model)

        # 2) Positional embeddings
        # Temporal positions: 0..Tin-1
        self.time_pos = nn.Parameter(torch.zeros(1, tin, 1, d_model))
        # Spatial positions: 0..S-1
        self.site_pos = nn.Parameter(torch.zeros(1, 1, num_sites, d_model))

        # 3) Temporal transformer encoder (applied per site)
        enc_layer_time = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer_time, num_layers=num_layers_time)

        # 4) Spatial transformer encoder (applied per timestep)
        enc_layer_space = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(enc_layer_space, num_layers=num_layers_space)

        # 5) Forecast head
        # We produce Tout * num_targets per site
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, tout * num_targets),
        )

        # Init parameters (simple, stable defaults)
        nn.init.normal_(self.time_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.site_pos, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (B, Tin, S, F)
        returns:
          y_hat: (B, Tout, S, num_targets)
          aux: dict (empty for now; later we will add attention weights)
        """

        B, Tin, S, F = x.shape
        assert Tin == self.tin, f"Expected Tin={self.tin}, got {Tin}"
        assert S == self.num_sites, f"Expected S={self.num_sites}, got {S}"
        assert F == self.num_features, f"Expected F={self.num_features}, got {F}"

        aux: Dict[str, torch.Tensor] = {}

        # Project + add positional embeddings
        h = self.in_proj(x)  # (B, Tin, S, d_model)
        h = h + self.time_pos + self.site_pos

        # ---- Temporal encoding (per site) ----
        # reshape to treat each site as a separate sequence
        # (B, Tin, S, d) -> (B*S, Tin, d)
        h_time = h.permute(0, 2, 1, 3).contiguous().view(B * S, Tin, self.d_model)
        h_time = self.temporal_encoder(h_time)  # (B*S, Tin, d)

        # take the last time step representation as summary
        h_last = h_time[:, -1, :]  # (B*S, d)
        h_last = h_last.view(B, S, self.d_model)  # (B, S, d)

        # ---- Spatial encoding (across sites) ----
        # Treat sites as tokens: (B, S, d)
        h_space = self.spatial_encoder(h_last)  # (B, S, d)

        # ---- Forecast head ----
        out = self.head(h_space)  # (B, S, Tout*num_targets)
        out = out.view(B, S, self.tout, self.num_targets)  # (B, S, Tout, targets)
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, Tout, S, targets)

        return out, aux