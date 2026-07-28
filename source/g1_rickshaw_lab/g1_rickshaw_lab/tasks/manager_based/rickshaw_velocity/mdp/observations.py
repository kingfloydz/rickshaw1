"""Fixed-schema actor and privileged observations."""

from __future__ import annotations

import torch

from g1_rickshaw_lab.policy_schema import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
)

TEACHER_STATIC_DOMAIN_DIM = TEACHER_STATIC_DIM

BASE_LINEAR_VELOCITY_SLICE = slice(0, 3)
BASE_ANGULAR_VELOCITY_SLICE = slice(3, 6)
PROJECTED_GRAVITY_SLICE = slice(6, 9)
COMMAND_SLICE = slice(9, 11)
JOINT_POSITION_SLICE = slice(11, 40)
JOINT_VELOCITY_SLICE = slice(40, 69)
PREVIOUS_ACTION_SLICE = slice(69, ACTOR_OBSERVATION_DIM)

BASE_LINEAR_VELOCITY_SCALE = 1.0
BASE_ANGULAR_VELOCITY_SCALE = 0.25
PROJECTED_GRAVITY_SCALE = 1.0
COMMAND_SCALE = 1.0
JOINT_POSITION_SCALE = 1.0
JOINT_VELOCITY_SCALE = 0.05
PREVIOUS_ACTION_SCALE = 1.0

# Unitree G1-29DoF velocity-policy sensor noise, expressed after this
# project's observation scaling.
ACTOR_OBSERVATION_NOISE_SCALE = (
    (0.5 * BASE_LINEAR_VELOCITY_SCALE,) * 3
    + (0.2 * BASE_ANGULAR_VELOCITY_SCALE,) * 3
    + (0.05 * PROJECTED_GRAVITY_SCALE,) * 3
    + (0.0,) * 2
    + (0.01 * JOINT_POSITION_SCALE,) * ACTION_DIM
    + (1.5 * JOINT_VELOCITY_SCALE,) * ACTION_DIM
    + (0.0,) * ACTION_DIM
)

TEACHER_STATIC_FEATURE_NAMES = (
    "robot.torso_mass",
    "cart.total_mass",
    "cart.com.x",
    "cart.com.y",
    "cart.com.z",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
    "terrain.slope",
)
TEACHER_DYNAMIC_FEATURE_NAMES = (
    "rickshaw.velocity.lin_vel_x",
    "rickshaw.velocity.ang_vel_z",
    "cart.pitch",
    "wheel.left_normal_force",
    "wheel.right_normal_force",
    *(f"hand.{side}.force.{axis}" for side in ("left", "right") for axis in ("s", "l", "n")),
)
if len(TEACHER_STATIC_FEATURE_NAMES) != TEACHER_STATIC_DIM:
    raise RuntimeError("teacher static feature schema has the wrong dimension")
if len(TEACHER_DYNAMIC_FEATURE_NAMES) != TEACHER_DYNAMIC_DIM:
    raise RuntimeError(f"teacher dynamic feature schema is not {TEACHER_DYNAMIC_DIM}-D")


def assemble_actor_observation(
    base_linear_velocity_b: torch.Tensor,
    base_angular_velocity_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    command: torch.Tensor,
    joint_position: torch.Tensor,
    q_ref: torch.Tensor,
    joint_velocity_value: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Assemble the 98-D deployment observation in policy order."""

    return torch.cat(
        (
            base_linear_velocity_b * BASE_LINEAR_VELOCITY_SCALE,
            base_angular_velocity_b * BASE_ANGULAR_VELOCITY_SCALE,
            projected_gravity_b * PROJECTED_GRAVITY_SCALE,
            command * COMMAND_SCALE,
            (joint_position - q_ref) * JOINT_POSITION_SCALE,
            joint_velocity_value * JOINT_VELOCITY_SCALE,
            previous_action * PREVIOUS_ACTION_SCALE,
        ),
        dim=-1,
    )


def assemble_teacher_dynamic_privilege(
    rickshaw_velocity: torch.Tensor,
    rickshaw_pitch: torch.Tensor,
    wheel_normal_force: torch.Tensor,
    per_hand_force_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_lateral_w: torch.Tensor,
    path_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Assemble per-step Rickshaw state and separate left/right hand forces."""

    hand_force_sln = torch.stack(
        tuple(
            torch.sum(per_hand_force_w * axis[:, None, :], dim=-1)
            for axis in (path_tangent_w, path_lateral_w, path_normal_w)
        ),
        dim=-1,
    )
    return torch.cat(
        (
            rickshaw_velocity,
            rickshaw_pitch[:, None],
            wheel_normal_force,
            hand_force_sln.flatten(start_dim=1),
        ),
        dim=-1,
    )


def normalize_features(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Normalize explicit feature bounds to [-1, 1]; singleton bounds map to zero."""

    expected = (values.shape[-1],)
    if lower.shape != expected or upper.shape != expected:
        raise ValueError(f"normalization bounds must both have shape {expected}")
    if not values.is_floating_point() or torch.any(~torch.isfinite(values)):
        raise ValueError("features to normalize must be finite floating-point values")
    lower = lower.to(device=values.device, dtype=values.dtype)
    upper = upper.to(device=values.device, dtype=values.dtype)
    if torch.any(~torch.isfinite(lower)) or torch.any(~torch.isfinite(upper)):
        raise ValueError("normalization bounds must be finite")
    width = upper - lower
    if torch.any(width < 0.0):
        raise ValueError("normalization bounds must be ordered")
    safe_width = torch.where(width > 0.0, width, torch.ones_like(width))
    normalized = 2.0 * (values - lower) / safe_width - 1.0
    normalized = torch.where(width > 0.0, normalized, torch.zeros_like(normalized))
    return torch.clamp(normalized, -1.0, 1.0)
