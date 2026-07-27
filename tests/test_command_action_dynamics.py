"""Pure-Torch tests for the active Mjlab command and dynamics kernels."""

from __future__ import annotations

import math

import pytest
import torch

from g1_rickshaw_lab.g1_motor_defaults import (
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    G1_ACTION_SCALE,
    G1_JOINT_EFFORT_LIMITS,
    G1_JOINT_STIFFNESS,
)
from g1_rickshaw_lab.policy_schema import ACTION_SCALE
from g1_rickshaw_lab.policy_schema import ACTION_DIM
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.dynamics import (
    RickshawKinematicState,
    connect_constraint_forces,
    quat_apply_wxyz,
    relative_position_in_yaw_frame,
    relative_yaw_from_quaternions,
    rolling_resistance_force,
    wheel_ground_frame,
    wheel_longitudinal_slip,
)


def test_reflected_inertia_matches_mjlab_1_5_3_g1_constants() -> None:
    assert ARMATURE_5020 == pytest.approx(0.003609725)
    assert ARMATURE_7520_14 == pytest.approx(0.01017752004132231)
    assert ARMATURE_7520_22 == pytest.approx(0.025101925)
    assert ARMATURE_4010 == pytest.approx(0.00425)


def test_action_scale_matches_unitree_motor_defaults() -> None:
    assert ACTION_DIM == 29
    expected = tuple(
        0.25 * effort / stiffness
        for effort, stiffness in zip(
            G1_JOINT_EFFORT_LIMITS, G1_JOINT_STIFFNESS, strict=True
        )
    )
    assert ACTION_SCALE == G1_ACTION_SCALE == expected
    torch.testing.assert_close(
        torch.tensor(ACTION_SCALE, dtype=torch.float64),
        torch.tensor(expected, dtype=torch.float64),
    )

def test_rolling_resistance_opposes_each_wheel() -> None:
    dtype = torch.float64
    tangent = torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=dtype)
    normal = torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=dtype)
    wheel_speed = torch.tensor([[1.0, 0.8], [-1.0, -0.7]], dtype=dtype)
    normal_force = torch.tensor([[310.0, 330.0], [280.0, 300.0]], dtype=dtype)
    c_rr = torch.tensor([0.02, 0.03], dtype=dtype)
    force = rolling_resistance_force(
        wheel_speed[..., None] * tangent[:, None, :],
        normal_force[..., None] * normal[:, None, :],
        tangent,
        normal,
        c_rr,
        velocity_epsilon=0.05,
    )
    force_s = torch.sum(force * tangent[:, None, :], dim=-1)
    expected = -c_rr[:, None] * normal_force * torch.tanh(wheel_speed / 0.05)
    torch.testing.assert_close(force_s, expected)
    assert torch.all(force_s * wheel_speed < 0.0)


def test_wheel_longitudinal_slip_is_contact_point_velocity() -> None:
    dtype = torch.float64
    center_velocity = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=dtype)
    angular_velocity = torch.tensor(
        [[[0.0, 1.0 / 0.3, 0.0], [0.0, 2.0 / 0.3, 0.0]]],
        dtype=dtype,
    )
    slip = wheel_longitudinal_slip(
        center_velocity,
        angular_velocity,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
        0.3,
    )
    torch.testing.assert_close(slip, torch.tensor([[0.0, -1.0]], dtype=dtype))


def test_rickshaw_kinematics_use_direct_policy_step_differences() -> None:
    dtype = torch.float64
    zeros = torch.zeros(1, dtype=dtype)
    state = RickshawKinematicState.initialized(zeros, torch.zeros((1, 3), dtype=dtype))
    state.update(
        torch.tensor([0.08], dtype=dtype),
        torch.tensor([[0.0, 0.2, 0.3]], dtype=dtype),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
        0.02,
    )
    torch.testing.assert_close(
        state.forward_acceleration, torch.tensor([4.0], dtype=dtype)
    )
    torch.testing.assert_close(
        state.pitch_angular_velocity, torch.tensor([0.2], dtype=dtype)
    )
    torch.testing.assert_close(
        state.pitch_angular_acceleration, torch.tensor([10.0], dtype=dtype)
    )
    torch.testing.assert_close(
        state.yaw_angular_acceleration, torch.tensor([15.0], dtype=dtype)
    )


def test_quaternion_vector_rotation_matches_mjlab() -> None:
    quat_apply = pytest.importorskip("mjlab.utils.lab_api.math").quat_apply

    generator = torch.Generator().manual_seed(7)
    quaternion = torch.randn((16, 4), dtype=torch.float64, generator=generator)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    vector = torch.randn((16, 3), dtype=torch.float64, generator=generator)
    torch.testing.assert_close(
        quat_apply_wxyz(quaternion, vector),
        quat_apply(quaternion, vector),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_wheel_ground_frame_uses_slope_normal_and_axle_direction() -> None:
    dtype = torch.float64
    slope = 0.3
    normal = torch.tensor([[-math.sin(slope), 0.0, math.cos(slope)]], dtype=dtype)
    forward, lateral, yaw_axis = wheel_ground_frame(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype), normal
    )
    torch.testing.assert_close(
        forward,
        torch.tensor([[math.cos(slope), 0.0, math.sin(slope)]], dtype=dtype),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(lateral, torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype))
    torch.testing.assert_close(yaw_axis, normal)


def test_relative_pose_is_invariant_to_common_world_yaw() -> None:
    dtype = torch.float64
    half_turn = 0.25 * math.pi
    yaw_90 = torch.tensor(
        [[math.cos(half_turn), 0.0, 0.0, math.sin(half_turn)]], dtype=dtype
    )
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    reset_relative = relative_position_in_yaw_frame(
        torch.zeros((1, 3), dtype=dtype),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype),
        identity,
    )
    turned_relative = relative_position_in_yaw_frame(
        torch.zeros((1, 3), dtype=dtype),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype),
        yaw_90,
    )
    torch.testing.assert_close(turned_relative, reset_relative, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        relative_yaw_from_quaternions(yaw_90, yaw_90),
        torch.zeros(1, dtype=dtype),
    )


def test_connect_constraint_forces_select_each_hand_and_ignore_padding() -> None:
    efc_type = torch.full((2, 12), 6, dtype=torch.long)
    efc_id = torch.full((2, 12), -1, dtype=torch.long)
    efc_force = torch.zeros((2, 12), dtype=torch.float64)

    efc_type[0, :6] = 0
    efc_id[0, :3] = 1
    efc_id[0, 3:6] = 0
    efc_force[0, :6] = torch.tensor([4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
    efc_type[0, 8:11] = 0
    efc_id[0, 8:11] = 0
    efc_force[0, 8:11] = 1000.0

    efc_type[1, 1:4] = 0
    efc_id[1, 1:4] = 0
    efc_force[1, 1:4] = torch.tensor([-1.0, -2.0, -3.0])
    efc_type[1, 5:8] = 0
    efc_id[1, 5:8] = 1
    efc_force[1, 5:8] = torch.tensor([-4.0, -5.0, -6.0])

    force = connect_constraint_forces(
        efc_type,
        efc_id,
        efc_force,
        torch.tensor([0, 1]),
        equality_constraint_type=0,
    )
    torch.testing.assert_close(
        force,
        torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            ],
            dtype=torch.float64,
        ),
    )
