"""Project-local MuJoCo/mjlab assets."""

from .g1_dex1 import get_g1_robot_cfg
from .rickshaw import get_rickshaw_cfg

__all__ = [
    "get_g1_robot_cfg",
    "get_rickshaw_cfg",
]
