"""Pure mechanical specification shared by geometry and simulation code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RickshawUrdfSpec:
    """Mechanical parameters shared by runtime and static-equilibrium code."""

    total_mass: float = 35.04
    center_of_mass: tuple[float, float, float] = (
        0.09037092351598175,
        0.0,
        0.5896865105424277,
    )
    wheel_radius: float = 0.3
    wheel_track: float = 0.756462
    hitch_x: float = 1.664929
    hitch_z: float = 0.105747
    hitch_half_width: float = 0.276


RICKSHAW_URDF_SPEC = RickshawUrdfSpec()
RICKSHAW_TOTAL_MASS = RICKSHAW_URDF_SPEC.total_mass
RICKSHAW_CENTER_OF_MASS = RICKSHAW_URDF_SPEC.center_of_mass
WHEEL_RADIUS = RICKSHAW_URDF_SPEC.wheel_radius
WHEEL_TRACK = RICKSHAW_URDF_SPEC.wheel_track
HITCH_X = RICKSHAW_URDF_SPEC.hitch_x
HITCH_Z = RICKSHAW_URDF_SPEC.hitch_z
HITCH_HALF_WIDTH = RICKSHAW_URDF_SPEC.hitch_half_width
HITCH_HEIGHT_RANGE = (0.75, 0.95)


__all__ = [
    "HITCH_HALF_WIDTH",
    "HITCH_HEIGHT_RANGE",
    "HITCH_X",
    "HITCH_Z",
    "RICKSHAW_CENTER_OF_MASS",
    "RICKSHAW_TOTAL_MASS",
    "RICKSHAW_URDF_SPEC",
    "RickshawUrdfSpec",
    "WHEEL_RADIUS",
    "WHEEL_TRACK",
]
