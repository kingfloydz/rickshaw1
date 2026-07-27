"""Per-environment inclined planes used by the rickshaw task."""

from __future__ import annotations

import math

import torch

from .sloped_reset import TERRAIN_SLOPES


def terrain_type_for_slope(slope: float) -> int:
    """Return the terrain type matching an explicitly selected slope."""

    if not math.isfinite(slope):
        raise ValueError("terrain slope must be finite")
    for terrain_type, configured_slope in enumerate(TERRAIN_SLOPES):
        if math.isclose(slope, configured_slope, abs_tol=1.0e-9):
            return terrain_type
    raise ValueError(f"terrain slope must be one of {TERRAIN_SLOPES}, got {slope}")


def assign_terrain_types(
    num_envs: int,
    *,
    device: torch.device | str,
    terrain_slope: float | None = None,
) -> torch.Tensor:
    """Distribute all slopes, or assign every environment to one selected slope."""

    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    if terrain_slope is not None:
        return torch.full_like(env_ids, terrain_type_for_slope(terrain_slope))
    return env_ids * len(TERRAIN_SLOPES) // num_envs


def terrain_plane_poses(
    env_origins: torch.Tensor,
    terrain_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world position and quaternion for each environment's terrain body."""

    slopes = torch.as_tensor(TERRAIN_SLOPES, device=terrain_types.device, dtype=env_origins.dtype)[terrain_types]
    half_angles = -0.5 * slopes
    quaternions = torch.zeros((terrain_types.numel(), 4), device=terrain_types.device, dtype=env_origins.dtype)
    quaternions[:, 0] = torch.cos(half_angles)
    quaternions[:, 2] = torch.sin(half_angles)
    return env_origins, quaternions


def terrain_frame(
    terrain_types: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tangent, lateral, and normal per environment."""

    slope_values = torch.as_tensor(TERRAIN_SLOPES, device=terrain_types.device, dtype=dtype)
    slopes = slope_values[terrain_types]
    zeros = torch.zeros_like(slopes)
    tangent = torch.stack((torch.cos(slopes), zeros, torch.sin(slopes)), dim=-1)
    lateral = torch.stack((zeros, torch.ones_like(slopes), zeros), dim=-1)
    normal = torch.stack((-torch.sin(slopes), zeros, torch.cos(slopes)), dim=-1)
    return tangent, lateral, normal


__all__ = [
    "TERRAIN_SLOPES",
    "assign_terrain_types",
    "terrain_frame",
    "terrain_plane_poses",
    "terrain_type_for_slope",
]
