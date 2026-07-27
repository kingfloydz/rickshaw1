"""Fixed-schema actor observations and the exclusive 61-frame history."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from g1_rickshaw_lab.policy_schema import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
    validate_history_length,
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
    "robot.velocity.s",
    "robot.velocity.l",
    "robot.velocity.n",
    "rickshaw.velocity.lin_vel_x",
    "rickshaw.velocity.ang_vel_z",
    "cart.velocity.l",
    "cart.velocity.n",
    "cart.pitch",
    "wheel.left_normal_force",
    "wheel.right_normal_force",
    *(f"hand.force.{axis}" for axis in ("s", "l", "n")),
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


@dataclass
class ObservationHistoryState:
    """History where ``current`` is never included in ``history``."""

    history: torch.Tensor | None
    current: torch.Tensor
    initialized: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        history_length: int = HISTORY_LENGTH,
        observation_dim: int = ACTOR_OBSERVATION_DIM,
        history_enabled: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ObservationHistoryState:
        history_length = validate_history_length(history_length)
        if observation_dim <= 0:
            raise ValueError("feature dimension must be positive")
        return cls(
            history=(
                torch.zeros(
                    (num_envs, history_length, observation_dim),
                    device=device,
                    dtype=dtype,
                )
                if history_enabled
                else None
            ),
            current=torch.zeros((num_envs, observation_dim), device=device, dtype=dtype),
            initialized=torch.zeros(num_envs, device=device, dtype=torch.bool),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        if self.history is not None:
            self.history[ids] = 0.0
        self.current[ids] = 0.0
        self.initialized[ids] = False

    def initialize(self, observation: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Fill all history frames with the first post-reset observation."""

        if env_ids is None:
            ids = torch.arange(self.current.shape[0], device=self.current.device)
        else:
            ids = env_ids.to(device=self.current.device, dtype=torch.long)
        if observation.shape == self.current.shape:
            observation = observation[ids]
        if observation.shape != (ids.numel(), self.current.shape[-1]):
            raise ValueError("initial history observation has the wrong shape")
        if self.history is not None:
            self.history[ids] = observation[:, None, :].expand(-1, self.history.shape[1], -1)
        self.current[ids] = observation
        self.initialized[ids] = True

    def advance(self, new_observation: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
        """Append old current, then replace current with the new observation."""

        if new_observation.shape != self.current.shape:
            raise ValueError("new_observation shape differs from history state")
        if valid_mask is None:
            valid_mask = torch.ones_like(self.initialized)
        if valid_mask.shape != self.initialized.shape:
            raise ValueError("valid_mask must have shape [N]")

        if self.history is None:
            self.current[valid_mask] = new_observation[valid_mask]
            self.initialized[valid_mask] = True
            return

        was_initialized = self.initialized.clone()
        initialize_mask = valid_mask & ~was_initialized
        advance_mask = valid_mask & was_initialized

        next_history = torch.cat((self.history[:, 1:], self.current[:, None, :]), dim=1)
        next_history[~advance_mask] = self.history[~advance_mask]
        initial = new_observation[initialize_mask]
        next_history[initialize_mask] = initial[:, None, :].expand(-1, next_history.shape[1], -1)
        self.history = next_history

        self.current[valid_mask] = new_observation[valid_mask]
        self.initialized[valid_mask] = True


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


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "BASE_LINEAR_VELOCITY_SLICE",
    "COMMAND_SLICE",
    "HISTORY_LENGTH",
    "TEACHER_DYNAMIC_FEATURE_NAMES",
    "TEACHER_STATIC_DIM",
    "TEACHER_STATIC_DOMAIN_DIM",
    "TEACHER_STATIC_FEATURE_NAMES",
    "ACTOR_OBSERVATION_NOISE_SCALE",
    "ObservationHistoryState",
    "assemble_actor_observation",
    "normalize_features",
]
