from __future__ import annotations

import pytest
import torch

from g1_rickshaw_lab.project_paths import PROJECT_ROOT
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.mimic import (
    JointMotionReference,
    joint_velocity_from_positions,
    load_joint_motion_reference,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.rewards import (
    mimic_joint_error_exp_value,
)


def test_hmr4d_reference_contains_7_1_seconds_of_leg_and_torso_motion() -> None:
    reference = load_joint_motion_reference(
        PROJECT_ROOT / "hmr4d_results_straight_g1.pkl",
        "cpu",
    )

    assert reference.fps == 30.0
    assert reference.joint_pos.shape == (213, 15)
    assert reference.joint_vel.shape == (213, 15)
    assert reference.duration_s == pytest.approx(7.1)
    assert reference.duration_steps(0.02) == 355


def test_joint_motion_reference_interpolates_position_and_velocity() -> None:
    joint_pos = torch.tensor([[0.0], [1.0], [4.0]])
    joint_vel = joint_velocity_from_positions(joint_pos, fps=1.0)
    reference = JointMotionReference(
        fps=1.0,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )

    sampled_pos, sampled_vel = reference.sample(torch.tensor([0.5, 1.5]))

    torch.testing.assert_close(sampled_pos, torch.tensor([[0.5], [2.5]]))
    torch.testing.assert_close(sampled_vel, torch.tensor([[1.5], [2.5]]))


def test_mimic_reward_is_zero_outside_mimic_commands() -> None:
    reward = mimic_joint_error_exp_value(
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
        torch.zeros((3, 2)),
        torch.tensor([True, False, True]),
        std=1.0,
    )

    torch.testing.assert_close(
        reward, torch.tensor([1.0, 0.0, torch.exp(torch.tensor(-1.0))])
    )
