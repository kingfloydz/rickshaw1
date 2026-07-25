"""Simulator-independent physical specifications for the rickshaw task."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass

from g1_rickshaw_lab.rickshaw_spec import HITCH_HALF_WIDTH, HITCH_X, HITCH_Z, WHEEL_RADIUS


@dataclass(kw_only=True)
class RickshawPoseTargetCfg:
    """Rickshaw front-lift target and reset acceptance tolerances."""

    wheel_radius: float = WHEEL_RADIUS
    hitch_x: float = HITCH_X
    hitch_z: float = HITCH_Z
    hitch_half_width: float = HITCH_HALF_WIDTH
    hitch_height_tolerance: float = MISSING
    hitch_vertical_speed_tolerance: float = MISSING


def target_pitch_from_hitch_height(target_height: float, cfg: RickshawPoseTargetCfg) -> float:
    """Solve the front-lift pitch that places the hitch at its target height."""

    vertical_offset = cfg.hitch_z - cfg.wheel_radius
    radius = math.hypot(cfg.hitch_x, vertical_offset)
    ratio = (target_height - cfg.wheel_radius) / radius
    if not -1.0 <= ratio <= 1.0:
        raise ValueError("rickshaw hitch-height target is unreachable")
    return math.asin(ratio) - math.atan2(vertical_offset, cfg.hitch_x)


__all__ = ["RickshawPoseTargetCfg", "target_pitch_from_hitch_height"]
