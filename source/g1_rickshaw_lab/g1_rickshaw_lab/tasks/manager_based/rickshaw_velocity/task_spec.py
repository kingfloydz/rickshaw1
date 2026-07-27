"""Simulator-independent physical specifications for the rickshaw task."""

from __future__ import annotations

import math

from g1_rickshaw_lab.rickshaw_spec import HITCH_X, HITCH_Z, WHEEL_RADIUS


def target_pitch_from_hitch_height(target_height: float) -> float:
    """Solve the front-lift pitch that places the hitch at its target height."""

    vertical_offset = HITCH_Z - WHEEL_RADIUS
    radius = math.hypot(HITCH_X, vertical_offset)
    ratio = (target_height - WHEEL_RADIUS) / radius
    if not -1.0 <= ratio <= 1.0:
        raise ValueError("rickshaw hitch-height target is unreachable")
    return math.asin(ratio) - math.atan2(vertical_offset, HITCH_X)


__all__ = ["target_pitch_from_hitch_height"]
