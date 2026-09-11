"""
StratoWatch 2.0 — Graph Spatio-Temporal Transformers.

Two research variants are implemented:

1. StaticGraphSTTransformer
   - Uses the supplied fixed spatial adjacency matrix.
   - No wind-conditioned modification.

2. DynamicWindGraphSTTransformer
   - Starts from the same fixed spatial adjacency.
   - Modulates edges using the latest observed wind direction.

Both models use the same temporal/spatial Transformer backbone
and differ ONLY in their graph propagation mechanism.

Contract:
    Input:
        x_raw: (B, Tin, S, F)

    Output:
        y_hat: (B, Tout, S, num_targets)

Important:
    These models perform DIRECT target prediction.

    They do NOT add O3_forecast / NO2_forecast to the output.

This is required because the Phase 7 multi-site artifact contains
direct scaled O3 and NO2 targets.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class _BaseGraphSTTransformer(nn.Module):
    """
    Shared implementation for static and dynamic Graph-ST models.
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

        self.num_features = num_features
        self.num_targets = num_targets
        self.tin = tin
        self.tout = tout
        self.num_sites = num_sites
        self.d_model = d_model

        # --------------------------------------------------
        # Feature projection
        # --------------------------------------------------

        self.in_proj = nn.Linear(
            num_features,
            d_model,
        )

        # --------------------------------------------------
        # Temporal Transformer
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
        # Spatial Transformer
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
        # Direct multi-horizon prediction head
        # --------------------------------------------------

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                tout * num_targets,
            ),
        )

    def _validate_inputs(
        self,
        x_raw: torch.Tensor,
        A_static: torch.Tensor,
        site_coords: Optional[torch.Tensor],
    ) -> Tuple[int, int, int, int]:
        if x_raw.ndim != 4:
            raise ValueError(
                "Expected x_raw shape "
                f"(B,T,S,F), got {tuple(x_raw.shape)}"
            )

        B, Tin, S, F = x_raw.shape

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

        if A_static.ndim != 2:
            raise ValueError(
                "Expected A_static shape (S,S), "
                f"got {tuple(A_static.shape)}"
            )

        if A_static.shape != (
            self.num_sites,
            self.num_sites,
        ):
            raise ValueError(
                "Expected A_static shape "
                f"({self.num_sites},{self.num_sites}), "
                f"got {tuple(A_static.shape)}"
            )

        if site_coords is not None:
            if site_coords.ndim != 2:
                raise ValueError(
                    "Expected site_coords shape (S,2), "
                    f"got {tuple(site_coords.shape)}"
                )

            if site_coords.shape != (
                self.num_sites,
                2,
            ):
                raise ValueError(
                    "Expected site_coords shape "
                    f"({self.num_sites},2), "
                    f"got {tuple(site_coords.shape)}"
                )

        return B, Tin, S, F

    def _encode_temporal(
        self,
        x_raw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            (B, S, d_model)
        """

        B, Tin, S, _ = x_raw.shape

        h = self.in_proj(x_raw)

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

        h_last = h_time[:, -1, :]

        return h_last.view(
            B,
            S,
            self.d_model,
        )

    def _build_static_adjacency(
        self,
        A_static: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Normalize fixed adjacency.

        Returns:
            (B-independent, S, S)
        """

        A = A_static.to(
            device=device,
            dtype=dtype,
        )

        row_sum = A.sum(
            dim=-1,
            keepdim=True,
        )

        A = A / (
            row_sum + 1e-8
        )

        return A

    def _propagate_graph(
        self,
        h: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Graph propagation.

        h:
            (B,S,d)

        adjacency:
            (S,S) or (B,S,S)
        """

        if adjacency.ndim == 2:
            return torch.matmul(
                adjacency.unsqueeze(0),
                h,
            )

        return torch.matmul(
            adjacency,
            h,
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        A_static: torch.Tensor,
        site_coords: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, Tin, S, F = self._validate_inputs(
            x_raw,
            A_static,
            site_coords,
        )

        # --------------------------------------------------
        # 1. Temporal modeling
        # --------------------------------------------------

        h = self._encode_temporal(
            x_raw
        )

        # --------------------------------------------------
        # 2. Graph propagation
        # --------------------------------------------------

        adjacency = self._get_graph_adjacency(
            x_raw=x_raw,
            A_static=A_static,
            site_coords=site_coords,
        )

        h_graph = self._propagate_graph(
            h,
            adjacency,
        )

        h = h + h_graph

        # --------------------------------------------------
        # 3. Spatial Transformer
        # --------------------------------------------------

        h = self.spatial_encoder(h)

        # --------------------------------------------------
        # 4. Direct multi-horizon prediction
        # --------------------------------------------------

        out = self.head(h)

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

        aux: Dict[str, torch.Tensor] = {}

        if return_attn:
            aux["graph_adjacency"] = adjacency

        return out, aux

    def _get_graph_adjacency(
        self,
        x_raw: torch.Tensor,
        A_static: torch.Tensor,
        site_coords: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class StaticGraphSTTransformer(
    _BaseGraphSTTransformer
):
    """
    Static Graph-ST Transformer.

    The graph is fixed for every batch and timestep.

    A_static is normalized row-wise and used directly.
    """

    def _get_graph_adjacency(
        self,
        x_raw: torch.Tensor,
        A_static: torch.Tensor,
        site_coords: Optional[torch.Tensor],
    ) -> torch.Tensor:

        return self._build_static_adjacency(
            A_static,
            dtype=x_raw.dtype,
            device=x_raw.device,
        )


class DynamicWindGraphSTTransformer(
    _BaseGraphSTTransformer
):
    """
    Dynamic Wind Graph-ST Transformer.

    The graph starts from the same static spatial adjacency,
    then edge strength is modulated using the latest observed
    horizontal wind direction.

    Wind feature indices:
        u_forecast = 8
        v_forecast = 9
    """

    def __init__(
        self,
        *args,
        wind_u_idx: int = 8,
        wind_v_idx: int = 9,
        graph_beta: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.wind_u_idx = wind_u_idx
        self.wind_v_idx = wind_v_idx
        self.graph_beta = graph_beta

    def _build_dynamic_adjacency(
        self,
        x_raw: torch.Tensor,
        A_static: torch.Tensor,
        site_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build wind-conditioned adjacency.

        x_raw:
            (B,T,S,F)

        A_static:
            (S,S)

        site_coords:
            (S,2)

        Returns:
            (B,S,S)
        """

        if site_coords is None:
            raise ValueError(
                "Dynamic wind graph requires site_coords."
            )

        B, _, S, _ = x_raw.shape

        # --------------------------------------------------
        # Latest observed wind vector per site
        # --------------------------------------------------

        wind = x_raw[
            :,
            -1,
            :,
            [
                self.wind_u_idx,
                self.wind_v_idx,
            ],
        ]

        # Normalize wind direction.
        wind_norm = torch.norm(
            wind,
            dim=-1,
            keepdim=True,
        )

        wind = wind / (
            wind_norm + 1e-6
        )

        # --------------------------------------------------
        # Direction from source site i to destination j
        # --------------------------------------------------

        diff = (
            site_coords.unsqueeze(0)
            - site_coords.unsqueeze(1)
        )

        distance = torch.norm(
            diff,
            dim=-1,
            keepdim=True,
        )

        direction = diff / (
            distance + 1e-6
        )

        # --------------------------------------------------
        # Wind / edge directional alignment
        # --------------------------------------------------

        alignment = (
            wind.unsqueeze(2)
            * direction.unsqueeze(0)
        ).sum(dim=-1)

        # Only downwind influence.
        alignment = torch.relu(
            alignment
        )

        # --------------------------------------------------
        # Modulate fixed adjacency
        # --------------------------------------------------

        A = A_static.to(
            device=x_raw.device,
            dtype=x_raw.dtype,
        )

        A_dyn = (
            A.unsqueeze(0)
            * (
                1.0
                + self.graph_beta
                * alignment
            )
        )

        # --------------------------------------------------
        # Row normalization
        # --------------------------------------------------

        A_dyn = A_dyn / (
            A_dyn.sum(
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )

        return A_dyn

    def _get_graph_adjacency(
        self,
        x_raw: torch.Tensor,
        A_static: torch.Tensor,
        site_coords: Optional[torch.Tensor],
    ) -> torch.Tensor:

        return self._build_dynamic_adjacency(
            x_raw=x_raw,
            A_static=A_static,
            site_coords=site_coords,
        )


# ----------------------------------------------------------
# Backward-compatible alias
# ----------------------------------------------------------

GraphSTTransformer = DynamicWindGraphSTTransformer