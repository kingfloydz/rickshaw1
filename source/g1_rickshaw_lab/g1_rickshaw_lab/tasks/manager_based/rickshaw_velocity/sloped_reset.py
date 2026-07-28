"""Slope-specific reset templates derived from the certified flat pose."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from g1_rickshaw_lab.configuration import G1_JOINT_ORDER
from g1_rickshaw_lab.rickshaw_spec import WHEEL_RADIUS

TERRAIN_SLOPES = tuple(index / 100.0 for index in range(-8, 11))


@dataclass(frozen=True)
class SlopedResetTemplates:
    robot_root_pose: np.ndarray
    cart_root_pose: np.ndarray
    robot_joint_position: np.ndarray


def _quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    half_angle = 0.5 * angle
    return np.concatenate(((np.cos(half_angle),), axis * np.sin(half_angle)))


def _quat_apply(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[0] * twice_cross + np.cross(xyz, twice_cross)


def _foot_contact_geoms(model: mujoco.MjModel) -> tuple[int, ...]:
    return tuple(
        geom_id
        for body_name in ("robot/left_ankle_roll_link", "robot/right_ankle_roll_link")
        for geom_id in range(
            int(model.body_geomadr[model.body(body_name).id]),
            int(model.body_geomadr[model.body(body_name).id] + model.body_geomnum[model.body(body_name).id]),
        )
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    )


def _bisect_cart_pitch(
    normal: np.ndarray,
    hitch_axis: np.ndarray,
    wheel_offsets: np.ndarray,
    target_height: float,
) -> float:
    def residual(angle: float) -> float:
        rotation = _axis_angle_quat(hitch_axis, angle)
        wheel_centers = np.stack([_quat_apply(rotation, offset) for offset in wheel_offsets])
        return float(np.mean(wheel_centers @ normal) - target_height)

    lower = -0.5
    upper = 0.5
    lower_residual = residual(lower)
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        middle_residual = residual(middle)
        if np.signbit(middle_residual) == np.signbit(lower_residual):
            lower = middle
            lower_residual = middle_residual
        else:
            upper = middle
    return 0.5 * (lower + upper)


def build_sloped_reset_templates(
    model: mujoco.MjModel,
    flat_qpos: np.ndarray,
    slopes: tuple[float, ...],
) -> SlopedResetTemplates:
    """Keep G1 upright, incline its ankles, and retain closed-chain wheel contact."""

    robot_root = int(model.joint("robot/floating_base_joint").qposadr[0])
    cart_root = int(model.joint("rickshaw/floating_base_joint").qposadr[0])
    ankle_qpos = tuple(int(model.joint(f"robot/{side}_ankle_pitch_joint").qposadr[0]) for side in ("left", "right"))
    robot_joint_qpos = tuple(int(model.joint(f"robot/{name}").qposadr[0]) for name in G1_JOINT_ORDER)
    foot_geoms = _foot_contact_geoms(model)
    wheel_body_ids = tuple(model.body(f"rickshaw/{side}_wheel_link").id for side in ("left", "right"))
    hitch_site_ids = tuple(model.site(f"rickshaw/{side}_hitch_site").id for side in ("left", "right"))

    flat_data = mujoco.MjData(model)
    flat_data.qpos[:] = flat_qpos
    mujoco.mj_forward(model, flat_data)
    flat_foot_clearance = float(
        np.mean([flat_data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0] for geom_id in foot_geoms])
    )
    flat_wheel_clearance = float(np.mean([flat_data.xpos[body_id, 2] - WHEEL_RADIUS for body_id in wheel_body_ids]))
    flat_hitch_positions = flat_data.site_xpos[list(hitch_site_ids)].copy()
    flat_hitch_center = np.mean(flat_hitch_positions, axis=0)
    hitch_axis = flat_hitch_positions[1] - flat_hitch_positions[0]
    hitch_axis /= np.linalg.norm(hitch_axis)
    flat_wheel_offsets = flat_data.xpos[list(wheel_body_ids)] - flat_hitch_center
    flat_robot_pose = flat_qpos[robot_root : robot_root + 7]
    flat_cart_pose = flat_qpos[cart_root : cart_root + 7]

    robot_poses: list[np.ndarray] = []
    cart_poses: list[np.ndarray] = []
    joint_positions: list[np.ndarray] = []
    data = mujoco.MjData(model)
    for slope in slopes:
        data.qpos[:] = flat_qpos
        data.qpos[list(ankle_qpos)] -= slope
        mujoco.mj_forward(model, data)
        normal = np.array((-np.sin(slope), 0.0, np.cos(slope)))
        foot_clearance = float(
            np.mean([normal @ data.geom_xpos[geom_id] - model.geom_size[geom_id, 0] for geom_id in foot_geoms])
        )
        root_height_offset = (flat_foot_clearance - foot_clearance) / normal[2]

        robot_pose = flat_robot_pose.copy()
        robot_pose[2] += root_height_offset
        target_hitch_center = flat_hitch_center + np.array((0.0, 0.0, root_height_offset))
        target_wheel_offset_height = WHEEL_RADIUS + flat_wheel_clearance - normal @ target_hitch_center
        cart_pitch = _bisect_cart_pitch(
            normal,
            hitch_axis,
            flat_wheel_offsets,
            target_wheel_offset_height,
        )
        cart_rotation = _axis_angle_quat(hitch_axis, cart_pitch)
        cart_pose = np.concatenate(
            (
                target_hitch_center + _quat_apply(cart_rotation, flat_cart_pose[:3] - flat_hitch_center),
                _quat_mul(cart_rotation, flat_cart_pose[3:7]),
            )
        )

        robot_poses.append(robot_pose)
        cart_poses.append(cart_pose)
        joint_positions.append(data.qpos[list(robot_joint_qpos)].copy())

    return SlopedResetTemplates(
        robot_root_pose=np.stack(robot_poses),
        cart_root_pose=np.stack(cart_poses),
        robot_joint_position=np.stack(joint_positions),
    )
