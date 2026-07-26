from __future__ import annotations

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards


def test_rickshaw_height_reward_kernels() -> None:
    height = torch.tensor([0.80, 0.82, 0.90])
    contact = torch.tensor([True, True, False])
    torch.testing.assert_close(
        rewards.hitch_height_exp_value(height, 0.80, contact),
        torch.tensor([1.0, torch.exp(torch.tensor(-1.0)), 0.0]),
    )
    torch.testing.assert_close(
        rewards.hitch_height_recovery_l2_value(height, 0.80),
        torch.tensor([0.0, 0.0, 1.0]),
    )


def test_linear_reward_ramp_reaches_target_at_iteration_1000() -> None:
    duration_steps = 1000 * 24
    assert rewards.linear_ramp_progress(0, duration_steps) == 0.0
    assert rewards.linear_ramp_progress(500 * 24, duration_steps) == 0.5
    assert rewards.linear_ramp_progress(1000 * 24, duration_steps) == 1.0
    assert rewards.linear_ramp_progress(2000 * 24, duration_steps) == 1.0


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
        rewards.peak_force_value(torch.tensor([[[30.0, 0.0, 0.0], [-30.0, 0.0, 0.0]]])),
        torch.tensor([0.125]),
    )


def test_reward_configuration_matches_mjlab_1_5_3_g1_flat() -> None:
    pytest.importorskip("mjlab")
    assert version("mjlab") == "1.5.3"

    from mjlab.tasks.velocity import mdp as velocity_mdp
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.env_cfg import (
        g1_rickshaw_env_cfg,
    )
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents.rsl_rl_cfg import (
        g1_rickshaw_teacher_ppo_runner_cfg,
    )
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mjlab_mdp

    cfg = g1_rickshaw_env_cfg()
    mjlab_cfg = unitree_g1_flat_env_cfg()
    official = {
        "track_linear_velocity": (mjlab_mdp.track_rickshaw_lin_vel_x, 2.0),
        "track_angular_velocity": (mjlab_mdp.track_rickshaw_ang_vel_z, 2.0),
        "upright": (velocity_mdp.upright, 1.0),
        "pose": (velocity_mdp.variable_posture, 1.0),
        "body_ang_vel": (velocity_mdp.body_angular_velocity_penalty, -0.05),
        "angular_momentum": (velocity_mdp.angular_momentum_penalty, -0.02),
        "dof_pos_limits": (velocity_mdp.joint_pos_limits, -1.0),
        "action_rate_l2": (velocity_mdp.action_rate_l2, -0.1),
        "air_time": (velocity_mdp.feet_air_time, 0.0),
        "foot_clearance": (velocity_mdp.feet_clearance, -2.0),
        "foot_swing_height": (velocity_mdp.feet_swing_height, -0.25),
        "foot_slip": (velocity_mdp.feet_slip, -0.1),
        "soft_landing": (velocity_mdp.soft_landing, -1.0e-5),
        "self_collisions": (velocity_mdp.self_collision_cost, -1.0),
    }
    rickshaw_penalties = {
        "rickshaw_forward_acceleration_l2": (
            mjlab_mdp.rickshaw_forward_acceleration_l2,
            -0.01,
        ),
        "rickshaw_pitch_angular_acceleration_l2": (
            mjlab_mdp.rickshaw_pitch_angular_acceleration_l2,
            -0.0025,
        ),
        "rickshaw_yaw_angular_acceleration_l2": (
            mjlab_mdp.rickshaw_yaw_angular_acceleration_l2,
            -0.0025,
        ),
        "rickshaw_pitch_angular_velocity_l2": (
            mjlab_mdp.rickshaw_pitch_angular_velocity_l2,
            -0.05,
        ),
        "rickshaw_wheel_slip_l2": (mjlab_mdp.rickshaw_wheel_slip_l2, -0.1),
    }
    relative_pose_and_force_penalties = {
        "rickshaw_g1_relative_position_l2": (
            mjlab_mdp.rickshaw_g1_relative_position_l2,
            -1.0,
            {"axle_weight": 4.0},
        ),
        "rickshaw_g1_relative_yaw_l2": (
            mjlab_mdp.rickshaw_g1_relative_yaw_l2,
            -0.5,
            {},
        ),
        "rickshaw_absolute_pitch_deviation_l2": (
            mjlab_mdp.rickshaw_absolute_pitch_deviation_l2,
            -0.5,
            {},
        ),
        "peak_force": (
            mjlab_mdp.peak_force,
            -4.0,
            {"soft_limit": 10.0, "hard_limit": 50.0},
        ),
    }
    assert set(cfg.rewards) == {
        *official,
        *rickshaw_penalties,
        *relative_pose_and_force_penalties,
        "hitch_height_exp",
        "hitch_height_recovery_l2",
    }
    for name, (func, weight) in official.items():
        term = cfg.rewards[name]
        mjlab_term = mjlab_cfg.rewards[name]
        assert term.func is func
        if name not in {"track_linear_velocity", "track_angular_velocity"}:
            assert term.func is mjlab_term.func
        assert term.weight == pytest.approx(0.1 if name == "upright" else weight)
        local_params = dict(term.params)
        mjlab_params = dict(mjlab_term.params)
        if name in {"track_linear_velocity", "track_angular_velocity"}:
            mjlab_params.pop("asset_cfg", None)
        if name == "pose":
            asset_cfg = local_params.pop("asset_cfg")
            assert asset_cfg.joint_names == (
                r".*_(hip|knee|ankle)_.*",
                r"waist_.*_joint",
            )
            mjlab_params.pop("asset_cfg")
            for key in ("std_walking", "std_running"):
                mjlab_params[key] = {
                    pattern: value
                    for pattern, value in mjlab_params[key].items()
                    if not any(
                        part in pattern for part in ("shoulder", "elbow", "wrist")
                    )
                }
        if name == "foot_swing_height":
            mjlab_params["target_height"] = 0.08
        assert local_params == mjlab_params

    for name, (func, weight) in rickshaw_penalties.items():
        assert cfg.rewards[name].func is func
        assert cfg.rewards[name].weight == pytest.approx(weight)
        assert cfg.rewards[name].params == {}

    for name, (func, weight, params) in relative_pose_and_force_penalties.items():
        assert cfg.rewards[name].func is func
        assert cfg.rewards[name].weight == pytest.approx(weight)
        assert cfg.rewards[name].params == params

    assert cfg.rewards["foot_swing_height"].params["target_height"] == 0.08
    assert cfg.rewards["foot_clearance"].params["target_height"] == 0.10
    assert cfg.rewards["hitch_height_exp"].weight == 0.5
    assert cfg.rewards["hitch_height_recovery_l2"].weight == -0.25
    assert cfg.commands["twist"].entity_name == "rickshaw"
    assert cfg.commands["twist"].ranges.lin_vel_y == (0.0, 0.0)
    assert cfg.commands["twist"].ranges.heading is None
    assert cfg.commands["twist"].viz.z_offset == 1.15
    assert cfg.scene.extent == 2.0
    assert cfg.viewer.distance == 3.0
    assert set(cfg.curriculum) == {"command_vel", "rickshaw_penalty_weights"}
    assert cfg.curriculum["command_vel"].params["velocity_stages"] == [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
        {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
        {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
    ]
    penalty_curriculum = cfg.curriculum["rickshaw_penalty_weights"]
    assert penalty_curriculum.func is mjlab_mdp.LinearRewardWeightCurriculum
    assert penalty_curriculum.params == {
        "reward_names": (
            *rickshaw_penalties,
            *relative_pose_and_force_penalties,
        ),
        "duration_steps": 1000 * 24,
    }
    target_weights = {
        name: cfg.rewards[name].weight
        for name in penalty_curriculum.params["reward_names"]
    }
    term_cfgs = {
        name: SimpleNamespace(weight=weight) for name, weight in target_weights.items()
    }
    curriculum_env = SimpleNamespace(
        common_step_counter=0,
        reward_manager=SimpleNamespace(
            get_term_cfg=lambda name: term_cfgs[name],
        ),
    )
    weight_curriculum = mjlab_mdp.LinearRewardWeightCurriculum(
        penalty_curriculum, curriculum_env
    )
    for step, progress in ((0, 0.0), (500 * 24, 0.5), (1000 * 24, 1.0)):
        curriculum_env.common_step_counter = step
        state = weight_curriculum(
            curriculum_env,
            torch.empty(0, dtype=torch.long),
            **penalty_curriculum.params,
        )
        assert state["progress"].item() == pytest.approx(progress)
        for name, target_weight in target_weights.items():
            assert term_cfgs[name].weight == pytest.approx(progress * target_weight)
    assert cfg.commands["twist"].resampling_time_range == (3.0, 8.0)
    assert not cfg.observations["critic_policy"].enable_corruption

    play_cfg = g1_rickshaw_env_cfg(play=True)
    assert play_cfg.episode_length_s == int(1e9)
    assert play_cfg.curriculum == {}
    assert play_cfg.commands["twist"].ranges.lin_vel_x == (-1.5, 2.0)
    assert play_cfg.commands["twist"].ranges.ang_vel_z == (-0.7, 0.7)

    agent = g1_rickshaw_teacher_ppo_runner_cfg()
    assert agent.num_steps_per_env == 24
    assert agent.max_iterations == 30_000
    assert agent.save_interval == 50
    assert agent.clip_actions is None
    assert agent.actor.hidden_dims == (512, 256, 128)
    assert agent.critic.hidden_dims == (512, 256, 128)
    assert agent.actor.obs_normalization
    assert agent.critic.obs_normalization
    assert agent.actor.distribution_cfg == {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    assert agent.algorithm.entropy_coef == 0.01
    assert agent.algorithm.num_mini_batches == 4
    assert agent.algorithm.learning_rate == 1.0e-3
    assert agent.algorithm.lam == 0.95
