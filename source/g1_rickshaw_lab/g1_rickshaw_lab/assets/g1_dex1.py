"""MuJoCo/mjlab asset for Unitree G1 with fixed Dex1 grippers."""

from __future__ import annotations

from copy import deepcopy

import mujoco

from g1_rickshaw_lab.g1_motor_defaults import (
    G1_ARTICULATION,
    G1_JOINT_ARMATURE,
    G1_JOINT_DAMPING,
    G1_JOINT_EFFORT_LIMITS,
    G1_JOINT_STIFFNESS,
    G1_MOTOR_PARAMETERS_BY_JOINT,
)
from g1_rickshaw_lab.project_paths import ASSET_ROOT

from .mujoco_spec import (
    GROUND_COLLISION_BIT,
    ROBOT_COLLISION_BIT,
    add_free_joint,
    load_urdf_spec,
    set_body_collision,
)

G1_DEX1_ASSET_DIR = ASSET_ROOT / "g1_dex1"
G1_DEX1_URDF_PATH = G1_DEX1_ASSET_DIR / "g1_29dof_mode_15_with_dex1_1.urdf"

GRASP_SITE_X = 0.11066269
GRASP_SITE_NAMES = ("left_grasp_site", "right_grasp_site")
FOOT_SITE_NAMES = ("left_foot", "right_foot")
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
GRIPPER_BODY_NAMES = (
    "left_dex1_base_link",
    "left_dex1_finger_link_1",
    "left_dex1_finger_link_2",
    "right_dex1_base_link",
    "right_dex1_finger_link_1",
    "right_dex1_finger_link_2",
)

G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}


def get_g1_spec() -> mujoco.MjSpec:
    """Build the floating G1 spec with task sites and sensors."""

    spec = load_urdf_spec(G1_DEX1_URDF_PATH)
    add_free_joint(spec, "pelvis")

    for joint_name, (_, _, effort_limit, _) in G1_MOTOR_PARAMETERS_BY_JOINT.items():
        joint = spec.joint(joint_name)
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.actfrcrange[:] = (-effort_limit, effort_limit)

    for geom in spec.geoms:
        # URDF collision geoms are group 0; group 1 contains render-only duplicates.
        if geom.group == 0:
            geom.contype = ROBOT_COLLISION_BIT
            geom.conaffinity = GROUND_COLLISION_BIT | ROBOT_COLLISION_BIT
            geom.condim = 1
        else:
            geom.contype = 0
            geom.conaffinity = 0
    for body_name in FOOT_BODY_NAMES:
        for geom in spec.body(body_name).geoms:
            if geom.group == 0 and geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                geom.condim = 3
                geom.priority = 1
                geom.friction = (0.6, 0.005, 0.0001)
    set_body_collision(
        spec,
        GRIPPER_BODY_NAMES,
        contype=0,
        conaffinity=0,
    )

    site_frames = (
        ("left_dex1_base_link", GRASP_SITE_NAMES[0], (0.7071067811865476, 0.7071067811865475, 0.0, 0.0)),
        ("right_dex1_base_link", GRASP_SITE_NAMES[1], (0.7071067811865476, -0.7071067811865475, 0.0, 0.0)),
    )
    for body_name, site_name, quat in site_frames:
        spec.body(body_name).add_site(
            name=site_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.006, 0.0, 0.0),
            pos=(GRASP_SITE_X, 0.0, 0.0),
            quat=quat,
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
    for body_name, site_name in zip(
        FOOT_BODY_NAMES,
        FOOT_SITE_NAMES,
        strict=True,
    ):
        spec.body(body_name).add_site(
            name=site_name,
            pos=(0.04, 0.0, -0.037),
            rgba=(1.0, 0.0, 0.0, 1.0),
        )
    spec.add_sensor(
        name="root_angmom",
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="pelvis",
    )
    return spec


def get_g1_robot_cfg():
    """Return a fresh mjlab EntityCfg; imports mjlab only when requested."""

    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=get_g1_spec,
        sort_actuators=True,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                **G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS,
                r".*_shoulder_pitch_joint": 0.35,
                r"left_shoulder_roll_joint": -0.08,
                r"right_shoulder_roll_joint": 0.08,
                r"left_elbow_joint": 0.22,
                r"right_elbow_joint": 0.22,
                r".*": 0.0,
            },
            joint_vel={r".*": 0.0},
        ),
        articulation=deepcopy(G1_ARTICULATION),
    )


__all__ = [
    "FOOT_SITE_NAMES",
    "G1_DEX1_URDF_PATH",
    "G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS",
    "G1_JOINT_ARMATURE",
    "G1_JOINT_DAMPING",
    "G1_JOINT_EFFORT_LIMITS",
    "G1_JOINT_STIFFNESS",
    "GRASP_SITE_NAMES",
    "get_g1_robot_cfg",
    "get_g1_spec",
]
