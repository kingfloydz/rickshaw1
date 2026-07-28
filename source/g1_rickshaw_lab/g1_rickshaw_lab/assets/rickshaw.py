"""MuJoCo/mjlab rickshaw asset and source-URDF validation."""

from __future__ import annotations

import mujoco

from ..project_paths import ASSET_ROOT
from ..rickshaw_spec import (
    HITCH_HALF_WIDTH,
    HITCH_X,
    HITCH_Z,
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
    WHEEL_RADIUS,
    WHEEL_TRACK,
)
from .mujoco_spec import (
    GROUND_COLLISION_BIT,
    RICKSHAW_COLLISION_BIT,
    add_free_joint,
    load_urdf_spec,
)

RICKSHAW_ASSET_DIR = ASSET_ROOT / "rickshaw"
RICKSHAW_URDF_PATH = RICKSHAW_ASSET_DIR / "rickshaw.urdf"

BASE_LINK_NAME = "base_link"
WHEEL_LINK_NAMES = ("left_wheel_link", "right_wheel_link")
WHEEL_JOINT_NAMES = ("left_wheel_joint", "right_wheel_joint")
HITCH_LINK_NAMES = ("left_tow_hitch_link", "right_tow_hitch_link")
HITCH_JOINT_NAMES = ("left_tow_hitch_joint", "right_tow_hitch_joint")
HITCH_SITE_NAMES = ("left_hitch_site", "right_hitch_site")


def get_rickshaw_spec() -> mujoco.MjSpec:
    """Build the passive two-wheel rickshaw and its two hitch sites."""

    spec = load_urdf_spec(RICKSHAW_URDF_PATH)
    spec.compiler.discardvisual = 0
    add_free_joint(spec, BASE_LINK_NAME)
    for geom in spec.geoms:
        geom.contype = RICKSHAW_COLLISION_BIT
        geom.conaffinity = GROUND_COLLISION_BIT
        geom.group = 3
    for geom in spec.body(BASE_LINK_NAME).geoms:
        geom.contype = 0
        geom.conaffinity = 0
    visual_meshes = (
        (
            BASE_LINK_NAME,
            "body_visual",
            "body.stl",
            (0.0001, 0.0001, 0.0001),
            (1.94034, -0.414504, -0.074999),
            (0.7071067811882787, 0.0, 0.0, 0.7071067811848163),
            (0.18, 0.004, 0.008, 1.0),
        ),
        (
            WHEEL_LINK_NAMES[0],
            "left_wheel_visual",
            "right_wheel.stl",
            (0.0001, 0.00008, 0.00008),
            (1.552272, -0.792735, -0.3),
            (0.7071067811882787, 0.0, 0.0, 0.7071067811848163),
            (0.05, 0.05, 0.05, 1.0),
        ),
        (
            WHEEL_LINK_NAMES[1],
            "right_wheel_visual",
            "left_wheel.stl",
            (0.0001, 0.00008, 0.00008),
            (1.552272, -0.036274, -0.3),
            (0.7071067811882787, 0.0, 0.0, 0.7071067811848163),
            (0.05, 0.05, 0.05, 1.0),
        ),
    )
    for body_name, name, filename, scale, pos, quat, rgba in visual_meshes:
        mesh = spec.add_mesh(name=f"{name}_mesh", file=filename, scale=scale)
        spec.body(body_name).add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            pos=pos,
            quat=quat,
            contype=0,
            conaffinity=0,
            group=1,
            rgba=rgba,
        )
    for body_name, site_name in zip(HITCH_LINK_NAMES, HITCH_SITE_NAMES, strict=True):
        spec.body(body_name).add_site(
            name=site_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.006, 0.0, 0.0),
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
    return spec


def get_rickshaw_cfg():
    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=get_rickshaw_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(-1.664929, 0.0, 0.0),
            joint_pos={r".*_wheel_joint": 0.0},
            joint_vel={r".*_wheel_joint": 0.0},
        ),
    )


__all__ = [
    "BASE_LINK_NAME",
    "HITCH_HALF_WIDTH",
    "HITCH_LINK_NAMES",
    "HITCH_SITE_NAMES",
    "HITCH_X",
    "HITCH_Z",
    "RICKSHAW_ASSET_DIR",
    "RICKSHAW_CENTER_OF_MASS",
    "RICKSHAW_TOTAL_MASS",
    "RICKSHAW_URDF_PATH",
    "WHEEL_JOINT_NAMES",
    "WHEEL_RADIUS",
    "WHEEL_TRACK",
    "get_rickshaw_cfg",
    "get_rickshaw_spec",
]
