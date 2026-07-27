from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from g1_rickshaw_lab.assets.g1_dex1 import (
    G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS,
    G1_DEX1_URDF_PATH,
    get_g1_spec,
    validate_g1_urdf,
)
from g1_rickshaw_lab.assets.mujoco_spec import GROUND_COLLISION_BIT, ROBOT_COLLISION_BIT
from g1_rickshaw_lab.assets.rickshaw import get_rickshaw_spec, validate_rickshaw_urdf
from g1_rickshaw_lab.rickshaw_spec import RICKSHAW_URDF_SPEC
from g1_rickshaw_lab.project_paths import PROJECT_ROOT
from g1_rickshaw_lab.static_equilibrium import (
    MujocoStaticEquilibrium,
    load_mujoco_static_equilibrium,
    save_mujoco_static_equilibrium,
    solve_fixed_contact_statics,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.closed_chain import (
    build_assembled_spec,
    validate_assembled_model,
)


def test_fixed_grippers_leave_exactly_29_robot_dofs() -> None:
    root = ET.parse(G1_DEX1_URDF_PATH).getroot()
    gripper_joints = [
        joint
        for joint in root.findall("joint")
        if "dex1_finger_joint" in joint.attrib["name"]
    ]
    assert len(gripper_joints) == 4
    assert all(joint.attrib["type"] == "fixed" for joint in gripper_joints)
    assert validate_g1_urdf() == ()
    model = get_g1_spec().compile()
    assert model.njnt == 30  # free base + 29 G1 joints


def test_rickshaw_has_0_6m_wheels_aligned_with_lowered_body() -> None:
    assert validate_rickshaw_urdf() == ()
    spec = RICKSHAW_URDF_SPEC
    assert spec.wheel_radius == 0.3
    assert spec.base_mass == 31.0
    assert spec.total_mass == 35.04
    assert spec.base_com_x == 0.1
    assert math.isclose(spec.body_vertical_offset, -(0.374999 - 0.3), abs_tol=1.0e-12)
    assert math.isclose(
        spec.base_com_x_before_shift - spec.base_com_x,
        0.6427393855133334,
        abs_tol=1.0e-12,
    )
    model = get_rickshaw_spec().compile()
    for name in ("left_wheel_link", "right_wheel_link"):
        geom_id = int(model.body_geomadr[model.body(name).id])
        assert math.isclose(model.geom_size[geom_id, 0] * 2.0, 0.6, abs_tol=1.0e-12)


def test_rickshaw_stl_geoms_use_a_default_visible_viewer_group() -> None:
    model = get_rickshaw_spec().compile()
    for name in ("body_visual", "left_wheel_visual", "right_wheel_visual"):
        geom = model.geom(name)
        assert geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH
        assert geom.group[0] == 1
        assert geom.contype[0] == 0
        assert geom.conaffinity[0] == 0


def test_g1_default_lower_body_matches_unitree_home_keyframe() -> None:
    expected = {
        "left_hip_pitch_joint": -0.1,
        "left_knee_joint": 0.3,
        "left_ankle_pitch_joint": -0.2,
        "right_hip_pitch_joint": -0.1,
        "right_knee_joint": 0.3,
        "right_ankle_pitch_joint": -0.2,
    }
    assert {name: G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS[name] for name in expected} == expected
    assert all(
        position == 0.0
        for name, position in G1_DEFAULT_LOWER_WAIST_JOINT_POSITIONS.items()
        if name not in expected
    )


def test_g1_uses_official_builtin_position_actuator_defaults() -> None:
    BuiltinPositionActuatorCfg = pytest.importorskip(
        "mjlab.actuator"
    ).BuiltinPositionActuatorCfg

    from g1_rickshaw_lab.assets.g1_dex1 import get_g1_robot_cfg
    from g1_rickshaw_lab.g1_motor_defaults import (
        ARMATURE_4010,
        ARMATURE_5020,
        ARMATURE_7520_14,
        ARMATURE_7520_22,
        DAMPING_4010,
        DAMPING_5020,
        DAMPING_7520_14,
        DAMPING_7520_22,
        DAMPING_RATIO,
        NATURAL_FREQUENCY,
        STIFFNESS_4010,
        STIFFNESS_5020,
        STIFFNESS_7520_14,
        STIFFNESS_7520_22,
    )

    actuators = get_g1_robot_cfg().articulation.actuators
    assert len(actuators) == 6
    assert all(
        isinstance(actuator, BuiltinPositionActuatorCfg) for actuator in actuators
    )
    assert math.isclose(NATURAL_FREQUENCY, 20.0 * 3.1415926535)
    assert DAMPING_RATIO == 2.0
    expected = (
        (STIFFNESS_5020, DAMPING_5020, 25.0, ARMATURE_5020),
        (STIFFNESS_7520_14, DAMPING_7520_14, 88.0, ARMATURE_7520_14),
        (STIFFNESS_7520_22, DAMPING_7520_22, 139.0, ARMATURE_7520_22),
        (STIFFNESS_4010, DAMPING_4010, 5.0, ARMATURE_4010),
        (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
        (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
    )
    for actuator, values in zip(actuators, expected, strict=True):
        actual = (
            actuator.stiffness,
            actuator.damping,
            actuator.effort_limit,
            actuator.armature,
        )
        assert all(
            math.isclose(value, target)
            for value, target in zip(actual, values, strict=True)
        )


def test_training_enables_mjlab_nan_guard() -> None:
    pytest.importorskip("mjlab")

    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.env_cfg import (
        g1_rickshaw_env_cfg,
    )

    training = g1_rickshaw_env_cfg(play=False)
    playback = g1_rickshaw_env_cfg(play=True)
    expected_dir = str(PROJECT_ROOT / "outputs" / "nan_dumps")
    assert training.sim.nan_guard.enabled
    assert training.sim.nan_guard.buffer_size == 100
    assert training.sim.nan_guard.max_envs_to_dump == 5
    assert training.sim.nan_guard.output_dir == expected_dir
    assert not playback.sim.nan_guard.enabled
    assert training.sim.mujoco.timestep == 0.005
    assert training.sim.mujoco.iterations == 10
    assert training.sim.mujoco.ls_iterations == 20
    assert training.sim.mujoco.ccd_iterations == 50
    assert training.sim.nconmax is None
    assert training.sim.njmax == 600
    assert training.sim.contact_sensor_maxmatch == 64
    assert training.decimation == 4
    assert training.scene.terrain.terrain_type == "plane"


def test_assembled_model_uses_two_connections_without_robot_rickshaw_collision() -> None:
    model = build_assembled_spec().compile()
    assert validate_assembled_model(model) == ()
    assert model.neq == 2
    assert model.nu == 29
    assert model.opt.timestep == 0.005
    assert model.opt.iterations == 10
    assert model.opt.ls_iterations == 20
    assert model.opt.ccd_iterations == 50
    np.testing.assert_allclose(model.eq_solref, np.tile((0.02, 1.0), (model.neq, 1)))
    body_names = [model.body(model.geom_bodyid[index]).name for index in range(model.ngeom)]
    robot_geoms = [index for index, name in enumerate(body_names) if name.startswith("robot/")]
    rickshaw_geoms = [index for index, name in enumerate(body_names) if name.startswith("rickshaw/")]
    gripper_geoms = [index for index, name in enumerate(body_names) if "_dex1_" in name]
    visual_geoms = [index for index in robot_geoms if model.geom_group[index] != 0]
    assert gripper_geoms
    assert visual_geoms
    assert all(
        model.geom_contype[index] == 0 and model.geom_conaffinity[index] == 0
        for index in gripper_geoms
    )
    assert all(
        model.geom_contype[index] == 0 and model.geom_conaffinity[index] == 0
        for index in visual_geoms
    )
    foot_collision_geoms = [
        index
        for index, name in enumerate(body_names)
        if name in ("robot/left_ankle_roll_link", "robot/right_ankle_roll_link")
        and model.geom_group[index] == 0
    ]
    assert len(foot_collision_geoms) == 8
    np.testing.assert_array_equal(model.geom_condim[foot_collision_geoms], 3)
    np.testing.assert_array_equal(model.geom_priority[foot_collision_geoms], 1)
    np.testing.assert_allclose(model.geom_friction[foot_collision_geoms, 0], 0.6)
    np.testing.assert_allclose(
        model.geom_solref[foot_collision_geoms],
        np.tile((0.02, 1.0), (len(foot_collision_geoms), 1)),
    )
    physical_geoms = [
        index for index in robot_geoms if index not in gripper_geoms and index not in visual_geoms
    ]
    assert all(model.geom_contype[index] == ROBOT_COLLISION_BIT for index in physical_geoms)
    assert all(
        model.geom_conaffinity[index] == (GROUND_COLLISION_BIT | ROBOT_COLLISION_BIT)
        for index in physical_geoms
    )
    nonfoot_collision_geoms = [index for index in physical_geoms if index not in foot_collision_geoms]
    np.testing.assert_array_equal(
        model.geom_condim[nonfoot_collision_geoms],
        np.ones(len(nonfoot_collision_geoms), dtype=int),
    )
    assert all(
        not (
            model.geom_contype[robot_geom] & model.geom_conaffinity[rickshaw_geom]
            or model.geom_contype[rickshaw_geom] & model.geom_conaffinity[robot_geom]
        )
        for robot_geom in robot_geoms
        for rickshaw_geom in rickshaw_geoms
    )


def test_static_hand_connections_are_force_only() -> None:
    solution = solve_fixed_contact_statics(
        mass=35.0,
        com_from_axle_sln=(0.1, 0.0, 0.5),
        handle_from_axle_sn=(1.5, 0.5),
        hitch_half_width=0.25,
        wheel_track=0.8,
    )
    assert all(len(force) == 3 for force in solution.handle_forces_sln)
    np.testing.assert_allclose(solution.cart_force_residual_sln, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.cart_moment_residual_sln, 0.0, atol=1.0e-12)


def test_static_rest_pose_is_bound_to_the_compiled_model(tmp_path) -> None:
    model = build_assembled_spec().compile()
    solution = MujocoStaticEquilibrium(
        qpos=np.zeros(model.nq),
        joint_actuator_torque=np.zeros(29),
        equality_position_error=0.0,
        support_height_error=0.0,
        hitch_height=0.85,
        acceleration_error=0.0,
        actuator_torque_ratio=0.0,
    )
    path = save_mujoco_static_equilibrium(model, solution, tmp_path / "rest.json")
    loaded = load_mujoco_static_equilibrium(model, path)
    np.testing.assert_array_equal(loaded.qpos, solution.qpos)

    mass = model.body_mass[model.body("rickshaw/base_link").id]
    model.body_mass[model.body("rickshaw/base_link").id] = np.nextafter(mass, np.inf)
    load_mujoco_static_equilibrium(model, path)

    model.body_mass[model.body("rickshaw/base_link").id] += 1.0
    with pytest.raises(ValueError, match="model signature"):
        load_mujoco_static_equilibrium(model, path)

    model = build_assembled_spec().compile()
    path = save_mujoco_static_equilibrium(model, solution, tmp_path / "physics-rest.json")
    model.opt.timestep = 0.002
    with pytest.raises(ValueError, match="model signature"):
        load_mujoco_static_equilibrium(model, path)
