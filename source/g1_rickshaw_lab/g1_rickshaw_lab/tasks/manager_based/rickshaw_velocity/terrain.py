"""Nineteen fixed inclined planes used by the rickshaw task."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

from .sloped_reset import TERRAIN_SLOPES


@dataclass(kw_only=True)
class InclinedPlanesCfg(SubTerrainCfg):
    """Generate one finite x-axis-inclined plane per configured angle."""

    angles: tuple[float, ...]
    thickness: float = 0.2
    _index: int = field(init=False, default=0)

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del difficulty, rng
        angle = self.angles[self._index]
        self._index += 1
        half_thickness = self.thickness / 2.0
        geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2.0, self.size[1] / 2.0, half_thickness),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, -half_thickness * math.cos(angle)),
            quat=(math.cos(-angle / 2.0), 0.0, math.sin(-angle / 2.0), 0.0),
            rgba=(0.35, 0.38, 0.42, 1.0),
        )
        return TerrainOutput(
            origin=np.array((self.size[0] / 2.0, self.size[1] / 2.0, 0.0)),
            geometries=[TerrainGeometry(geom=geom)],
        )


def make_sloped_terrain_cfg() -> TerrainGeneratorCfg:
    """Create one 100 m by 100 m patch for every configured slope."""

    return TerrainGeneratorCfg(
        size=(100.0, 100.0),
        num_rows=1,
        num_cols=len(TERRAIN_SLOPES),
        sub_terrains={"slopes": InclinedPlanesCfg(angles=TERRAIN_SLOPES)},
        color_scheme="none",
        add_lights=True,
    )


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
    "InclinedPlanesCfg",
    "TERRAIN_SLOPES",
    "make_sloped_terrain_cfg",
    "terrain_frame",
]
