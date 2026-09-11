"""
StratoWatch 2.0 — Baseline Spatio-Temporal Transformer.

Direct-target multi-horizon model.

Contract:
    Input:  (B, Tin, S, F)
    Output: (B, Tout, S, num_targets)

This model intentionally contains NO explicit graph propagation
and NO forecast-baseline reconstruction.

It is the non-graph baseline against which the two Graph-ST
models are compared.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


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

        # --------------------------------------------------
        # 1. Feature projection
        # --------------------------------------------------

        self.in_proj = nn.Linear(
            num_features,
            d_model,
        )

        # --------------------------------------------------
        # 2. Learnable temporal and spatial embeddings
        # --------------------------------------------------

        self.time_pos = nn.Parameter(
            torch.zeros(1, tin, 1, d_model)
        )

        self.site_pos = nn.Parameter(
            torch.zeros(1, 1, num_sites, d_model)
        )

        nn.init.normal_(
            self.time_pos,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.site_pos,
            mean=0.0,
            std=0.02,
        )

        # --------------------------------------------------
        # 3. Temporal Transformer
        # --------------------------------------------------

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_layers_time,
        )

        # --------------------------------------------------
        # 4. Spatial Transformer
        # --------------------------------------------------

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer,
            num_layers=num_layers_space,
        )

        # --------------------------------------------------
        # 5. Direct multi-horizon prediction head
        # --------------------------------------------------

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                tout * num_targets,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x:
            Input tensor of shape (B, Tin, S, F).

        x_mask:
            Optional input-validity mask.
            Accepted for interface compatibility.
            Currently not used for attention masking.

        return_attn:
            Reserved for future attention extraction.

        Returns
        -------
        y_hat:
            Direct target predictions with shape
            (B, Tout, S, num_targets).

        aux:
            Auxiliary outputs dictionary.
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected x shape (B,T,S,F), got {tuple(x.shape)}"
            )

        B, Tin, S, F = x.shape

        if Tin != self.tin:
            raise ValueError(
                f"Expected Tin={self.tin}, got {Tin}"
            )

        if S != self.num_sites:
            raise ValueError(
                f"Expected S={self.num_sites}, got {S}"
            )

        if F != self.num_features:
            raise ValueError(
                f"Expected F={self.num_features}, got {F}"
            )

        # --------------------------------------------------
        # 1. Feature projection + positional information
        # --------------------------------------------------

        h = self.in_proj(x)

        h = (
            h
            + self.time_pos
            + self.site_pos
        )

        # --------------------------------------------------
        # 2. Temporal modeling independently per site
        # --------------------------------------------------

        h_time = (
            h.permute(0, 2, 1, 3)
            .contiguous()
            .view(
                B * S,
                Tin,
                self.d_model,
            )
        )

        h_time = self.temporal_encoder(
            h_time
        )

        # Use the latest temporal representation.
        h_last = h_time[:, -1, :]

        h_last = h_last.view(
            B,
            S,
            self.d_model,
        )

        # --------------------------------------------------
        # 3. Spatial modeling across sites
        # --------------------------------------------------

        h_space = self.spatial_encoder(
            h_last
        )

        # --------------------------------------------------
        # 4. Direct multi-horizon prediction
        # --------------------------------------------------

        out = self.head(h_space)

        out = out.view(
            B,
            S,
            self.tout,
            self.num_targets,
        )

        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
        )

        # IMPORTANT:
        # No baseline reconstruction here.
        # Y in the Phase 7 multi-site artifact contains
        # direct scaled target values.

        aux: Dict[str, torch.Tensor] = {}

        return out, aux