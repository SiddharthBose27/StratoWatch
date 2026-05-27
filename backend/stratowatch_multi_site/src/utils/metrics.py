"""
src/utils/metrics.py

Masked regression metrics for multi-horizon, multi-site forecasting.
We use Y_mask to ignore missing targets (if any).
"""

from __future__ import annotations
import torch


def masked_mae(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    y_hat, y, mask: (B, Tout, S, C)
    mask is 1 where valid, 0 where invalid
    """
    diff = torch.abs(y_hat - y) * mask
    denom = torch.clamp(mask.sum(), min=eps)
    return diff.sum() / denom


def masked_mse(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = (y_hat - y) ** 2 * mask
    denom = torch.clamp(mask.sum(), min=eps)
    return diff.sum() / denom


def masked_rmse(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(masked_mse(y_hat, y, mask, eps=eps))

def masked_huber(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Masked Huber (SmoothL1) loss.
    y_hat, y, mask: (B, Tout, S, C)
    """
    diff = (y_hat - y).abs()
    # Huber formula
    loss = torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    loss = loss * mask
    denom = torch.clamp(mask.sum(), min=eps)
    return loss.sum() / denom