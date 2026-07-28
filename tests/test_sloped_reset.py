from __future__ import annotations

import mujoco
import numpy as np

from g1_rickshaw_lab.configuration import G1_JOINT_ORDER
from g1_rickshaw_lab.rickshaw_spec import WHEEL_RADIUS
from g1_rickshaw_lab.static_equilibrium import load_mujoco_static_equilibrium
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.closed_chain import (
    build_assembled_spec,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.sloped_reset import (
    TERRAIN_SLOPES,
    build_sloped_reset_templates,
)


def test_sloped_reset_keeps_g1_root_upright_and_only_changes_ankle_pitch() -> None:
    model = build_assembled_spec(with_ground=True).compile()
    flat = load_mujoco_static_equilibrium(model).qpos
    templates = build_sloped_reset_templates(model, flat, TERRAIN_SLOPES)
    robot_root = int(model.joint("robot/floating_base_joint").qposadr[0])
    flat_joint_position = np.array(
        [flat[int(model.joint(f"robot/{name}").qposadr[0])] for name in G1_JOINT_ORDER]
    )

    flat_root_quaternion = np.broadcast_to(
        flat[robot_root + 3 : robot_root + 7], (len(TERRAIN_SLOPES), 4)
    )
    np.testing.assert_allclose(templates.robot_root_pose[:, 3:7], flat_root_quaternion)
    flat_root_xy = np.broadcast_to(
        flat[robot_root : robot_root + 2], (len(TERRAIN_SLOPES), 2)
    )
    np.testing.assert_allclose(templates.robot_root_pose[:, :2], flat_root_xy)
    assert (
        np.max(np.abs(templates.robot_root_pose[:, 2] - flat[robot_root + 2])) < 1.1e-3
    )
    for joint_index, joint_name in enumerate(G1_JOINT_ORDER):
        expected = np.repeat(flat_joint_position[joint_index], len(TERRAIN_SLOPES))
        if joint_name.endswith("ankle_pitch_joint"):
            expected -= TERRAIN_SLOPES
        np.testing.assert_allclose(
            templates.robot_joint_position[:, joint_index], expected
        )


def test_sloped_reset_aligns_feet_and_wheels_without_breaking_hitches() -> None:
    model = build_assembled_spec(with_ground=True).compile()
    flat = load_mujoco_static_equilibrium(model).qpos
    templates = build_sloped_reset_templates(model, flat, TERRAIN_SLOPES)
    robot_root = int(model.joint("robot/floating_base_joint").qposadr[0])
    cart_root = int(model.joint("rickshaw/floating_base_joint").qposadr[0])
    joint_qpos = tuple(
        int(model.joint(f"robot/{name}").qposadr[0]) for name in G1_JOINT_ORDER
    )
    foot_geoms = tuple(
        geom_id
        for body_name in ("robot/left_ankle_roll_link", "robot/right_ankle_roll_link")
        for geom_id in range(
            int(model.body_geomadr[model.body(body_name).id]),
            int(
                model.body_geomadr[model.body(body_name).id]
                + model.body_geomnum[model.body(body_name).id]
            ),
        )
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    )
    wheel_bodies = tuple(
        model.body(f"rickshaw/{side}_wheel_link").id for side in ("left", "right")
    )

    flat_data = mujoco.MjData(model)
    flat_data.qpos[:] = flat
    mujoco.mj_forward(model, flat_data)
    flat_foot_clearance = np.mean(
        [
            flat_data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0]
            for geom_id in foot_geoms
        ]
    )
    flat_wheel_clearance = np.mean(
        [flat_data.xpos[body_id, 2] - WHEEL_RADIUS for body_id in wheel_bodies]
    )

    for slope_index, slope in enumerate(TERRAIN_SLOPES):
        data = mujoco.MjData(model)
        data.qpos[:] = flat
        data.qpos[robot_root : robot_root + 7] = templates.robot_root_pose[slope_index]
        data.qpos[cart_root : cart_root + 7] = templates.cart_root_pose[slope_index]
        data.qpos[list(joint_qpos)] = templates.robot_joint_position[slope_index]
        mujoco.mj_forward(model, data)
        normal = np.array((-np.sin(slope), 0.0, np.cos(slope)))
        foot_clearance = np.array(
            [
                normal @ data.geom_xpos[geom_id] - model.geom_size[geom_id, 0]
                for geom_id in foot_geoms
            ]
        )
        wheel_clearance = np.array(
            [normal @ data.xpos[body_id] - WHEEL_RADIUS for body_id in wheel_bodies]
        )
        hitch_error = max(
            np.linalg.norm(
                data.site(f"robot/{side}_grasp_site").xpos
                - data.site(f"rickshaw/{side}_hitch_site").xpos
            )
            for side in ("left", "right")
        )

        np.testing.assert_allclose(
            np.mean(foot_clearance), flat_foot_clearance, atol=1.0e-10
        )
        np.testing.assert_allclose(
            np.mean(wheel_clearance), flat_wheel_clearance, atol=1.0e-10
        )
        assert np.ptp(foot_clearance) < 7.0e-5
        assert np.ptp(wheel_clearance) < 5.0e-5
        assert hitch_error < 2.0e-4
