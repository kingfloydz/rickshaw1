"""Project-local MuJoCo/mjlab assets."""

from .g1_dex1 import get_g1_robot_cfg, validate_g1_urdf
from .rickshaw import (
    RICKSHAW_URDF_SPEC,
    get_rickshaw_cfg,
    validate_rickshaw_urdf,
)

__all__ = [
    "RICKSHAW_URDF_SPEC",
    "get_g1_robot_cfg",
    "get_rickshaw_cfg",
    "validate_g1_urdf",
    "validate_rickshaw_urdf",
]
