"""Simulator-independent state and sampling kernels used by Mjlab events."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from typing import Any

import torch

from g1_rickshaw_lab.assets.rickshaw import (
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
)


@dataclass
class RickshawRuntimeState:
    wheel_normal_force: torch.Tensor
    hitch_height: torch.Tensor
    hitch_vertical_speed: torch.Tensor
    pitch: torch.Tensor
    two_wheel_contact: torch.Tensor
    connection_residual: torch.Tensor
    hand_force_w: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        num_wheels: int = 2,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> RickshawRuntimeState:
        scalar = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(
            wheel_normal_force=torch.zeros(
                (num_envs, num_wheels), device=device, dtype=dtype
            ),
            hitch_height=scalar.clone(),
            hitch_vertical_speed=scalar.clone(),
            pitch=scalar.clone(),
            two_wheel_contact=torch.zeros(
                num_envs, device=device, dtype=torch.bool
            ),
            connection_residual=scalar.clone(),
            hand_force_w=torch.zeros((num_envs, 3), device=device, dtype=dtype),
        )


@dataclass
class StabilityState:
    theta_fat: torch.Tensor
    fat_valid: torch.Tensor
    fat_force_consistent: torch.Tensor
    fat_force_relative_error: torch.Tensor
    torso_pitch: torch.Tensor
    zmp_s: torch.Tensor
    zmp_margin: torch.Tensor
    zmp_valid: torch.Tensor
    ground_reaction_normal: torch.Tensor
    support_center_w: torch.Tensor
    support_points_sy: torch.Tensor
    support_point_mask: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> StabilityState:
        scalar = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(
            theta_fat=scalar.clone(),
            fat_valid=torch.zeros(num_envs, device=device, dtype=torch.bool),
            fat_force_consistent=torch.zeros(
                num_envs, device=device, dtype=torch.bool
            ),
            fat_force_relative_error=torch.zeros(
                (num_envs, 2), device=device, dtype=dtype
            ),
            torso_pitch=scalar.clone(),
            zmp_s=scalar.clone(),
            zmp_margin=scalar.clone(),
            zmp_valid=torch.zeros(num_envs, device=device, dtype=torch.bool),
            ground_reaction_normal=scalar.clone(),
            support_center_w=torch.zeros((num_envs, 3), device=device, dtype=dtype),
            support_points_sy=torch.zeros(
                (num_envs, 8, 2), device=device, dtype=dtype
            ),
            support_point_mask=torch.zeros(
                (num_envs, 8), device=device, dtype=torch.bool
            ),
        )


DOMAIN_RANDOMIZATION_NAMES = (
    "torso.mass_delta",
    "payload.mass",
    "payload.com.x",
    "payload.com.y",
    "payload.com.z",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
)
DOMAIN_PARAMETER_NAMES = DOMAIN_RANDOMIZATION_NAMES


@dataclass(kw_only=True)
class DomainRandomizationCfg:
    enabled: bool = True
    ranges: Mapping[str, tuple[float, float]] = MISSING
    nominal: Mapping[str, float] = MISSING
    calibration: Mapping[str, Any] = MISSING

    def validate(self) -> None:
        required = set(DOMAIN_PARAMETER_NAMES)
        for label, values in (("ranges", self.ranges), ("nominal", self.nominal)):
            if not isinstance(values, Mapping) or set(values) != required:
                raise ValueError(
                    f"domain randomization {label} must contain exactly {sorted(required)}"
                )
        for name, interval in self.ranges.items():
            low, high = map(float, interval)
            nominal = float(self.nominal[name])
            if not all(map(math.isfinite, (low, high, nominal))) or not (
                low <= nominal <= high
            ):
                raise ValueError(f"invalid range or nominal value for {name!r}")
        for name in (
            "rolling_resistance.c_rr",
            "wheel.left_damping",
            "wheel.right_damping",
        ):
            if float(self.ranges[name][0]) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if float(self.ranges["terrain.friction"][0]) <= 0.0:
            raise ValueError("terrain friction must stay positive")
        if not isinstance(self.calibration, Mapping):
            raise ValueError("domain randomization calibration must be a mapping")


def sample_domain_parameters(
    cfg: DomainRandomizationCfg,
    batch_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    cfg.validate()
    if batch_size < 0:
        raise ValueError("batch_size cannot be negative")

    def sample(name: str) -> torch.Tensor:
        if not cfg.enabled:
            return torch.full(
                (batch_size,), float(cfg.nominal[name]), device=device, dtype=dtype
            )
        low, high = map(float, cfg.ranges[name])
        if low == high:
            return torch.full((batch_size,), low, device=device, dtype=dtype)
        return torch.empty((batch_size,), device=device, dtype=dtype).uniform_(
            low, high, generator=generator
        )

    return {name: sample(name) for name in DOMAIN_RANDOMIZATION_NAMES}


def effective_cart_mass_com_bounds(
    ranges: Mapping[str, tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mass_low, mass_high = map(float, ranges["payload.mass"])
    lower = [RICKSHAW_TOTAL_MASS + mass_low]
    upper = [RICKSHAW_TOTAL_MASS + mass_high]
    for axis, name in enumerate(
        ("payload.com.x", "payload.com.y", "payload.com.z")
    ):
        payload_low, payload_high = map(float, ranges[name])
        candidates = [
            (
                RICKSHAW_TOTAL_MASS * RICKSHAW_CENTER_OF_MASS[axis]
                + payload_mass * payload_com
            )
            / (RICKSHAW_TOTAL_MASS + payload_mass)
            for payload_mass in (mass_low, mass_high)
            for payload_com in (payload_low, payload_high)
        ]
        lower.append(min(candidates))
        upper.append(max(candidates))
    return tuple(lower), tuple(upper)


def _update_teacher_static_domain(
    env: Any,
    cfg: DomainRandomizationCfg,
    sampled: Mapping[str, torch.Tensor],
) -> None:
    raw = torch.cat(
        (
            env.effective_torso_mass[:, None],
            env.effective_cart_mass_com,
            sampled["rolling_resistance.c_rr"][:, None],
            sampled["terrain.friction"][:, None],
            torch.stack(
                (
                    sampled["wheel.left_damping"],
                    sampled["wheel.right_damping"],
                ),
                dim=-1,
            ),
        ),
        dim=-1,
    )
    cart_lower, cart_upper = effective_cart_mass_com_bounds(cfg.ranges)
    nominal_torso_mass = float(
        env._default_robot_masses_cpu[0, env.torso_body_id]
    )
    lower = torch.tensor(
        (
            nominal_torso_mass + cfg.ranges["torso.mass_delta"][0],
            *cart_lower,
            cfg.ranges["rolling_resistance.c_rr"][0],
            cfg.ranges["terrain.friction"][0],
            cfg.ranges["wheel.left_damping"][0],
            cfg.ranges["wheel.right_damping"][0],
        ),
        device=env.device,
        dtype=raw.dtype,
    )
    upper = torch.tensor(
        (
            nominal_torso_mass + cfg.ranges["torso.mass_delta"][1],
            *cart_upper,
            cfg.ranges["rolling_resistance.c_rr"][1],
            cfg.ranges["terrain.friction"][1],
            cfg.ranges["wheel.left_damping"][1],
            cfg.ranges["wheel.right_damping"][1],
        ),
        device=env.device,
        dtype=raw.dtype,
    )
    from .observations import TEACHER_STATIC_DOMAIN_DIM, normalize_features

    if raw.shape != (env.num_envs, TEACHER_STATIC_DOMAIN_DIM):
        raise RuntimeError(
            f"effective teacher static domain must have shape [N,{TEACHER_STATIC_DOMAIN_DIM}]"
        )
    env.teacher_static_domain_raw = raw
    env.teacher_static_domain_bounds = (lower, upper)
    env.normalized_teacher_static_domain = normalize_features(raw, lower, upper)


__all__ = [
    "DOMAIN_PARAMETER_NAMES",
    "DOMAIN_RANDOMIZATION_NAMES",
    "DomainRandomizationCfg",
    "RickshawRuntimeState",
    "StabilityState",
    "effective_cart_mass_com_bounds",
    "sample_domain_parameters",
]
