from __future__ import annotations

import pytest
import torch

pytest.importorskip("mjlab")

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.terrain import (
    TERRAIN_SLOPES,
    InclinedPlanesCfg,
    make_sloped_terrain_cfg,
    terrain_frame,
)


def test_terrain_has_the_reference_nineteen_slopes() -> None:
    assert TERRAIN_SLOPES == tuple(index / 100.0 for index in range(-8, 11))
    cfg = make_sloped_terrain_cfg()
    assert cfg.num_rows == 1
    assert cfg.num_cols == 19
    assert cfg.size == (100.0, 100.0)
    slopes = cfg.sub_terrains["slopes"]
    assert isinstance(slopes, InclinedPlanesCfg)
    assert slopes.angles == TERRAIN_SLOPES


def test_terrain_frame_matches_each_inclined_plane() -> None:
    terrain_types = torch.arange(19)
    tangent, lateral, normal = terrain_frame(terrain_types, dtype=torch.float64)
    torch.testing.assert_close(lateral, torch.eye(3, dtype=torch.float64)[1].expand(19, -1))
    torch.testing.assert_close(torch.sum(tangent * normal, dim=-1), torch.zeros(19, dtype=torch.float64))
