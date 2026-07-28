from __future__ import annotations

import pytest
import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards


def test_rickshaw_height_recovery_reward_kernel() -> None:
    height = torch.tensor([0.80, 0.82, 0.90])
    torch.testing.assert_close(
        rewards.hitch_height_recovery_l2_value(height, 0.80),
        torch.tensor([0.0, 0.0, 1.0]),
    )


def test_wheel_slip_reward_only_penalizes_contacting_wheels() -> None:
    torch.testing.assert_close(
        rewards.wheel_slip_l2_value(
            torch.tensor([[0.5, -2.0], [1.0, 3.0]]),
            torch.tensor([[True, False], [True, True]]),
        ),
        torch.tensor([0.25, 10.0]),
    )


def test_reset_relative_pose_and_per_hand_peak_force_penalties() -> None:
    reference_position = torch.tensor([1.0, 0.0, 0.8])
    torch.testing.assert_close(
        rewards.relative_position_l2_value(
            reference_position[None], reference_position
        ),
        torch.zeros(1),
    )
    torch.testing.assert_close(
        rewards.relative_position_l2_value(
            torch.tensor([[1.1, 0.1, 0.9]]), reference_position
        ),
        torch.tensor([0.06]),
    )
    torch.testing.assert_close(
        rewards.angle_deviation_l2_value(
            torch.tensor([-torch.pi + 0.1]), torch.pi - 0.1
        ),
        torch.tensor([0.04]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rewards.peak_force_value(
            torch.tensor([[[30.0, 0.0, 0.0], [-30.0, 0.0, 0.0]]])
        ),
        torch.tensor([0.125]),
    )


def test_rickshaw_reward_and_command_contract() -> None:
    pytest.importorskip("mjlab")

    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mjlab_mdp
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.env_cfg import (
        g1_rickshaw_env_cfg,
    )

    cfg = g1_rickshaw_env_cfg()
    expected = {
        "track_linear_velocity": (mjlab_mdp.track_rickshaw_lin_vel_x, 2.0),
        "track_angular_velocity": (mjlab_mdp.track_rickshaw_ang_vel_z, 2.0),
        "rickshaw_forward_acceleration_l2": (
            mjlab_mdp.rickshaw_forward_acceleration_l2,
            -0.05,
        ),
        "rickshaw_pitch_angular_acceleration_l2": (
            mjlab_mdp.rickshaw_pitch_angular_acceleration_l2,
            -0.01,
        ),
        "rickshaw_yaw_angular_acceleration_l2": (
            mjlab_mdp.rickshaw_yaw_angular_acceleration_l2,
            -0.01,
        ),
        "rickshaw_pitch_angular_velocity_l2": (
            mjlab_mdp.rickshaw_pitch_angular_velocity_l2,
            -1.0,
        ),
        "rickshaw_wheel_slip_l2": (mjlab_mdp.rickshaw_wheel_slip_l2, -0.1),
        "rickshaw_g1_relative_position_l2": (
            mjlab_mdp.rickshaw_g1_relative_position_l2,
            -4.0,
        ),
        "rickshaw_g1_relative_yaw_l2": (
            mjlab_mdp.rickshaw_g1_relative_yaw_l2,
            -0.6,
        ),
        "rickshaw_absolute_pitch_deviation_l2": (
            mjlab_mdp.rickshaw_absolute_pitch_deviation_l2,
            -0.5,
        ),
        "peak_force": (mjlab_mdp.peak_force, -3.0),
        "hitch_height_recovery_l2": (mjlab_mdp.hitch_height_recovery_l2, -0.25),
    }
    for name, (function, weight) in expected.items():
        assert cfg.rewards[name].func is function
        assert cfg.rewards[name].weight == pytest.approx(weight)

    command = cfg.commands["twist"]
    assert command.entity_name == "rickshaw"
    assert command.ranges.lin_vel_x == (-1.5, 2.0)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (-0.7, 0.7)

    play_cfg = g1_rickshaw_env_cfg(play=True)
    assert play_cfg.episode_length_s == int(1e9)
    assert play_cfg.curriculum == {}
