"""Rickshaw-specific reward kernels."""

from __future__ import annotations

import torch

HITCH_HEIGHT_ERROR_SCALE_M = 0.02
HITCH_HEIGHT_RECOVERY_DEADBAND_M = 0.05
HITCH_HEIGHT_RECOVERY_SCALE_M = 0.05


def hitch_height_exp_value(
    hitch_height: torch.Tensor,
    target_height: float,
    two_wheel_contact: torch.Tensor,
    *,
    std: float = HITCH_HEIGHT_ERROR_SCALE_M,
) -> torch.Tensor:
    return torch.exp(-torch.square((hitch_height - target_height) / std)) * two_wheel_contact.to(
        hitch_height.dtype
    )


def hitch_height_recovery_l2_value(
    hitch_height: torch.Tensor,
    target_height: float,
    *,
    deadband: float = HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    scale: float = HITCH_HEIGHT_RECOVERY_SCALE_M,
) -> torch.Tensor:
    normalized = torch.relu(torch.abs(hitch_height - target_height) - deadband) / scale
    return torch.where(
        normalized <= 1.0,
        torch.square(normalized),
        2.0 * normalized - 1.0,
    )


__all__ = [
    "HITCH_HEIGHT_ERROR_SCALE_M",
    "HITCH_HEIGHT_RECOVERY_DEADBAND_M",
    "HITCH_HEIGHT_RECOVERY_SCALE_M",
    "hitch_height_exp_value",
    "hitch_height_recovery_l2_value",
]
