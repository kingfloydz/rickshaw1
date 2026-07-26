"""Rickshaw-specific reward kernels."""

from __future__ import annotations

import torch

HITCH_HEIGHT_ERROR_SCALE_M = 0.02
HITCH_HEIGHT_RECOVERY_DEADBAND_M = 0.05
HITCH_HEIGHT_RECOVERY_SCALE_M = 0.05


def mimic_joint_error_exp_value(
    joint_state: torch.Tensor,
    reference_state: torch.Tensor,
    active: torch.Tensor,
    std: float,
) -> torch.Tensor:
    error = torch.mean(torch.square(joint_state - reference_state), dim=-1)
    return torch.exp(-error / std**2) * active.to(joint_state.dtype)


def stepped_ramp_progress(
    step: int,
    interval_steps: int,
    duration_steps: int,
) -> float:
    completed_steps = (step // interval_steps) * interval_steps
    return min(float(completed_steps) / duration_steps, 1.0)


def hitch_height_exp_value(
    hitch_height: torch.Tensor,
    target_height: float,
    two_wheel_contact: torch.Tensor,
    *,
    std: float = HITCH_HEIGHT_ERROR_SCALE_M,
) -> torch.Tensor:
    return torch.exp(-torch.square((hitch_height - target_height) / std)) * two_wheel_contact.to(hitch_height.dtype)


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


def wheel_slip_l2_value(
    longitudinal_slip: torch.Tensor,
    wheel_contact: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(
        torch.square(longitudinal_slip) * wheel_contact.to(longitudinal_slip.dtype),
        dim=-1,
    )


def relative_position_l2_value(
    relative_position_b: torch.Tensor,
    reference_position_b: torch.Tensor,
    *,
    axle_weight: float = 4.0,
) -> torch.Tensor:
    """Penalize robot-cart translation drift, emphasizing the axle direction."""

    error = relative_position_b - reference_position_b
    return torch.square(error[..., 0]) + axle_weight * torch.square(error[..., 1]) + torch.square(error[..., 2])


def angle_deviation_l2_value(angle: torch.Tensor, reference_angle: torch.Tensor | float) -> torch.Tensor:
    """Squared wrapped angular deviation from a reset reference."""

    error = angle - reference_angle
    wrapped = torch.atan2(torch.sin(error), torch.cos(error))
    return torch.square(wrapped)


def peak_force_value(
    per_hand_force_w: torch.Tensor,
    *,
    soft_limit: float = 10.0,
    hard_limit: float = 50.0,
) -> torch.Tensor:
    """Cubic soft-limit penalty on the larger individual hand force."""

    force_peak = torch.linalg.vector_norm(per_hand_force_w, dim=-1).amax(dim=-1)
    return torch.pow(torch.relu(force_peak - soft_limit) / (hard_limit - soft_limit), 3)


__all__ = [
    "HITCH_HEIGHT_ERROR_SCALE_M",
    "HITCH_HEIGHT_RECOVERY_DEADBAND_M",
    "HITCH_HEIGHT_RECOVERY_SCALE_M",
    "angle_deviation_l2_value",
    "hitch_height_exp_value",
    "hitch_height_recovery_l2_value",
    "mimic_joint_error_exp_value",
    "peak_force_value",
    "relative_position_l2_value",
    "stepped_ramp_progress",
    "wheel_slip_l2_value",
]
