from __future__ import annotations
from ctypes import alignment
import torch
import torch.nn as nn
from typing import Optional


class GraphConv(nn.Module):
    """
    Simple graph propagation layer:
    H' = A_hat H W
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x_raw: torch.Tensor, A_static: torch.Tensor, site_coords: torch.Tensor):
        """
        x_raw: (B, Tin, S, F)
        A_static: (S, S)
        site_coords: (S, 2)
        """

        B, Tin, S, F = x_raw.shape

        # Save wind from last timestep BEFORE projection
        # We need indices of u_forecast and v_forecast
        # These depend on feature order — we will hardcode for now:
        # In your dataset:
        # index of u_forecast = 8
        # index of v_forecast = 9

        u_idx = 8
        v_idx = 9

        wind = x_raw[:, -1, :, [u_idx, v_idx]]  # (B,S,2)
        wind = wind / (torch.norm(wind, dim=-1, keepdim=True) + 1e-6)  # unit wind direction

        # ---- Project features ----
        x = self.in_proj(x_raw)

        # ---- Temporal encoder ----
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * S, Tin, -1)
        x = self.temporal_encoder(x)
        x = x[:, -1, :]
        x = x.reshape(B, S, -1)

        # ---- Compute dynamic adjacency ----

        # site_coords: (S,2)
        # compute direction vectors d_ij
        diff = site_coords.unsqueeze(0) - site_coords.unsqueeze(1)  # (S,S,2)
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-6
        direction = diff / dist  # normalized direction

        # wind alignment
        # wind: (B,S,2)
        wind_exp = wind.unsqueeze(2)  # (B,S,1,2)
        direction_exp = direction.unsqueeze(0)  # (1,S,S,2)

        alignment = (wind_exp * direction_exp).sum(dim=-1)  # (B,S,S)
        alignment = torch.relu(alignment)

        beta = 0.5
        A_dyn = A_static.unsqueeze(0) * (1 + beta * alignment)

        # row normalize
        A_dyn = A_dyn / (A_dyn.sum(dim=-1, keepdim=True) + 1e-8)

        # ---- Graph propagation ----
        x_graph = torch.matmul(A_dyn, x)
        x = x + 0.5 * x_graph

        # ---- Spatial encoder ----
        x = self.spatial_encoder(x)

        # ---- Output ----
        out = self.head(x)
        out = out.view(B, S, self.tout, -1)
        out = out.permute(0, 2, 1, 3)

        return out


class GraphSTTransformer(nn.Module):
    """
    Graph-Enhanced Spatio-Temporal Transformer

    Pipeline:
    1) Feature projection
    2) Temporal Transformer (per site)
    3) Graph propagation (across sites)
    4) Spatial Transformer
    5) Forecast head
    """

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

        self.tout = tout
        self.num_sites = num_sites

        # Input projection
        self.in_proj = nn.Linear(num_features, d_model)

        # Temporal encoder
        encoder_layer_time = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer_time,
            num_layers=num_layers_time,
        )

        # Graph layer
        self.graph_layer = GraphConv(d_model)

        # Spatial encoder
        encoder_layer_space = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            encoder_layer_space,
            num_layers=num_layers_space,
        )

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, tout * num_targets),
        )

    def forward(self, x_raw: torch.Tensor, A_static: torch.Tensor, site_coords: torch.Tensor):
        """
        Dynamic wind-conditioned Graph-ST forward pass.

        rgs:
           x_raw: (B, Tin, S, F) raw input features (NOT projected)
           A_static: (S, S) base adjacency (distance/corr hybrid)
            site_coords: (S, 2) [lat, lon] per site

        Returns:
            out: (B, Tout, S, 2) predictions for O3 and NO2
        """
        B, Tin, S, F = x_raw.shape

        # Your feature ordering earlier:
        # 08 = u_forecast, 09 = v_forecast
        u_idx = 8
        v_idx = 9

        # Wind vector at last timestep for each site: (B, S, 2)
        wind = x_raw[:, -1, :, [u_idx, v_idx]]

        # ---- Project features ----
        x = self.in_proj(x_raw)  # (B, Tin, S, D)

        # ---- Temporal modeling (per site) ----
        x = x.permute(0, 2, 1, 3)      # (B, S, Tin, D)
        x = x.reshape(B * S, Tin, -1)  # (B*S, Tin, D)
        x = self.temporal_encoder(x)
        x = x[:, -1, :]                # (B*S, D)
        x = x.reshape(B, S, -1)        # (B, S, D)

        # ---- Build dynamic wind-conditioned adjacency A_dyn (B, S, S) ----
        # Direction from site i to site j: (S, S, 2)
        # (j - i) so positive dot means wind at i points toward j
        diff = site_coords.unsqueeze(0) - site_coords.unsqueeze(1)  # (S, S, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-6
        direction = diff / dist  # (S, S, 2)

        # Alignment: (B, S, S)
        alignment = (wind.unsqueeze(2) * direction.unsqueeze(0)).sum(dim=-1)
        alignment = torch.relu(alignment)  # only downwind influence

        beta = 0.5
        A_dyn = A_static.unsqueeze(0) * (1.0 + beta * alignment)  # (B, S, S)

        # Row-normalize (stochastic)
        A_dyn = A_dyn / (A_dyn.sum(dim=-1, keepdim=True) + 1e-8)

        # ---- Graph propagation (residual) ----
        x_graph = torch.matmul(A_dyn, x)  # (B, S, D)
        x = x + 0.5 * x_graph

        # ---- Spatial modeling (attention across sites) ----
        x = self.spatial_encoder(x)  # (B, S, D)

        # ---- Forecast head ----
        out = self.head(x)  # (B, S, Tout*2)
        out = out.view(B, S, self.tout, -1)
        out = out.permute(0, 2, 1, 3)  # (B, Tout, S, 2)

        return out