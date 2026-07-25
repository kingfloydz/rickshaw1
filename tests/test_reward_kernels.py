from __future__ import annotations

from importlib.metadata import version

import pytest
import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.observations import (
    GAIT_PERIOD_S,
)


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
    assert GAIT_PERIOD_S == 1.0


def test_reward_configuration_matches_mjlab_1_5_3_g1_flat() -> None:
    pytest.importorskip("mjlab")
    assert version("mjlab") == "1.5.3"

    from mjlab.tasks.velocity import mdp as velocity_mdp
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.env_cfg import (
        g1_rickshaw_env_cfg,
    )

    cfg = g1_rickshaw_env_cfg()
    mjlab_cfg = unitree_g1_flat_env_cfg()
    official = {
        "track_linear_velocity": (velocity_mdp.track_linear_velocity, 2.0),
        "track_angular_velocity": (velocity_mdp.track_angular_velocity, 2.0),
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
    assert set(cfg.rewards) == {
        *official,
        "hitch_height_exp",
        "hitch_height_recovery_l2",
    }
    for name, (func, weight) in official.items():
        term = cfg.rewards[name]
        mjlab_term = mjlab_cfg.rewards[name]
        assert term.func is func
        assert term.func is mjlab_term.func
        assert term.weight == pytest.approx(0.2 if name == "upright" else weight)
        local_params = dict(term.params)
        mjlab_params = dict(mjlab_term.params)
        if name in {"track_linear_velocity", "track_angular_velocity"}:
            assert local_params.pop("asset_cfg").name == "rickshaw"
        if name == "foot_swing_height":
            mjlab_params["target_height"] = 0.08
        assert local_params == mjlab_params

    assert cfg.rewards["foot_swing_height"].params["target_height"] == 0.08
    assert cfg.rewards["foot_clearance"].params["target_height"] == 0.10
    assert cfg.rewards["hitch_height_exp"].weight == 0.5
    assert cfg.rewards["hitch_height_recovery_l2"].weight == -0.25
    assert cfg.commands["twist"].entity_name == "rickshaw"
