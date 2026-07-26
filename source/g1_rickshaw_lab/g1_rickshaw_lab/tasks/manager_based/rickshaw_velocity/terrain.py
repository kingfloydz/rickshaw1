"""Per-environment inclined planes used by the rickshaw task."""

from __future__ import annotations

import torch

from .sloped_reset import TERRAIN_SLOPES


def assign_terrain_types(num_envs: int, *, device: torch.device | str) -> torch.Tensor:
    """Distribute environments evenly across the nineteen configured slopes."""

    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
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
]
