"""MuJoCo site-connect closed chain for the fixed grippers and rickshaw."""

from __future__ import annotations

import mujoco

from g1_rickshaw_lab.assets.g1_dex1 import (
    GRASP_SITE_NAMES,
    get_g1_robot_cfg,
)
from g1_rickshaw_lab.assets.mujoco_spec import (
    ALL_COLLISION_BITS,
    GROUND_COLLISION_BIT,
)
from g1_rickshaw_lab.assets.rickshaw import HITCH_SITE_NAMES, get_rickshaw_spec

ROBOT_ENTITY_NAME = "robot"
RICKSHAW_ENTITY_NAME = "rickshaw"
CONNECTION_NAMES = ("left_grasp_connection", "right_grasp_connection")


def add_closed_chain_constraints(spec: mujoco.MjSpec) -> None:
    """Connect both fixed gripper centers to the rickshaw crossbar."""

    for side, grasp_site, hitch_site in zip(("left", "right"), GRASP_SITE_NAMES, HITCH_SITE_NAMES, strict=True):
        name1 = f"{ROBOT_ENTITY_NAME}/{grasp_site}"
        name2 = f"{RICKSHAW_ENTITY_NAME}/{hitch_site}"
        if spec.site(name1) is None or spec.site(name2) is None:
            raise ValueError(f"missing {side} closed-chain sites: {name1}, {name2}")
        spec.add_equality(
            name=f"{side}_grasp_connection",
            type=mujoco.mjtEq.mjEQ_CONNECT,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=name1,
            name2=name2,
            active=1,
        )


def build_assembled_spec(*, with_ground: bool = True) -> mujoco.MjSpec:
    """Build a standalone one-environment model for validation/statics."""

    from mjlab.entity import Entity

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.005
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.ccd_iterations = 50
    if with_ground:
        ground = spec.worldbody.add_geom(
            name="terrain",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0.0, 0.0, 0.05),
        )
        ground.contype = GROUND_COLLISION_BIT
        ground.conaffinity = ALL_COLLISION_BITS
        ground.friction[:3] = (1.0, 0.005, 0.0001)
    robot_spec = Entity(get_g1_robot_cfg()).spec
    spec.attach(robot_spec, prefix=f"{ROBOT_ENTITY_NAME}/", frame=spec.worldbody.add_frame())
    spec.attach(get_rickshaw_spec(), prefix=f"{RICKSHAW_ENTITY_NAME}/", frame=spec.worldbody.add_frame())
    add_closed_chain_constraints(spec)
    return spec


__all__ = [
    "ROBOT_ENTITY_NAME",
    "RICKSHAW_ENTITY_NAME",
    "CONNECTION_NAMES",
    "add_closed_chain_constraints",
    "build_assembled_spec",
]
