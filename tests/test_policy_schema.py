"""Cross-layer checks for the single policy ABI contract."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from g1_rickshaw_lab import policy_schema
from g1_rickshaw_lab.rl import actor_critic, context_encoder, teacher_model
from g1_rickshaw_lab.rl.rsl_rl_models import _DeploymentController
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import observations
from g1_rickshaw_lab.training_contract import (
    TRAINING_CONFIGURATION_KEY,
    _deployment_contract,
)


def test_policy_dimensions_are_shared_across_runtime_layers() -> None:
    assert context_encoder.OBSERVATION_DIM == policy_schema.ACTOR_OBSERVATION_DIM
    assert context_encoder.HISTORY_LENGTH == policy_schema.HISTORY_LENGTH
    assert policy_schema.DEFAULT_CONTEXT_DIM == 16
    assert policy_schema.SUPPORTED_CONTEXT_DIMS == (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32)
    assert actor_critic.CURRENT_OBSERVATION_DIM == policy_schema.ACTOR_OBSERVATION_DIM
    assert actor_critic.ACTION_DIM == policy_schema.ACTION_DIM
    assert actor_critic.CRITIC_PRIVILEGE_DIM == policy_schema.CRITIC_PRIVILEGED_DIM
    assert teacher_model.DYNAMIC_PRIVILEGE_DIM == policy_schema.TEACHER_DYNAMIC_DIM
    assert teacher_model.STATIC_PRIVILEGE_DIM == policy_schema.TEACHER_STATIC_DIM
    assert observations.ACTOR_OBSERVATION_DIM == policy_schema.ACTOR_OBSERVATION_DIM
    assert observations.HISTORY_LENGTH == policy_schema.HISTORY_LENGTH
    assert observations.TEACHER_DYNAMIC_DIM == policy_schema.TEACHER_DYNAMIC_DIM
    assert observations.TEACHER_STATIC_DIM == policy_schema.TEACHER_STATIC_DIM

    for latent_dim in policy_schema.SUPPORTED_CONTEXT_DIMS:
        assert policy_schema.validate_context_dim(latent_dim) == latent_dim
    with pytest.raises(ValueError, match="context dimension"):
        policy_schema.validate_context_dim(8.0)


def test_direct_action_contract_is_shared_by_simulation_and_deployment() -> None:
    class Policy(nn.Module):
        def forward(self, current: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            del history
            return current[:, : policy_schema.ACTION_DIM]

    expected_scale = torch.tensor(policy_schema.ACTION_SCALE)
    controller = _DeploymentController(Policy())
    torch.testing.assert_close(controller.action_scale, expected_scale)
    current = torch.ones(2, policy_schema.ACTOR_OBSERVATION_DIM)
    history = torch.zeros(2, 61, policy_schema.ACTOR_OBSERVATION_DIM)
    q_ref = torch.zeros(2, policy_schema.ACTION_DIM)
    action, target = controller(current, history, q_ref)
    torch.testing.assert_close(action, torch.ones_like(action))
    torch.testing.assert_close(target, expected_scale.expand_as(target))


def test_deployment_manifest_uses_the_policy_schema() -> None:
    manifest = _deployment_contract(
        {
            TRAINING_CONFIGURATION_KEY: {
                "stage": "s2_student_ppo",
                "training_parameters": {
                    "rollout_steps": 24,
                    "latent_dim": 24,
                    "history_length": 61,
                },
            }
        }
    )

    assert manifest["policy"]["inputs"] == {
        "current": [None, policy_schema.ACTOR_OBSERVATION_DIM],
        "history": [
            None,
            policy_schema.HISTORY_LENGTH,
            policy_schema.ACTOR_OBSERVATION_DIM,
        ],
    }
    assert manifest["policy"]["context_dim"] == 24
    assert manifest["policy"]["output"]["normalized_action"] == [
        None,
        policy_schema.ACTION_DIM,
    ]
    assert manifest["action"]["scale_rad_per_normalized_action"] == list(
        policy_schema.ACTION_SCALE
    )
    assert max(abs(value) for value in manifest["action"]["q_ref"]) <= 2.0
    assert manifest["action"]["mapping"] == "joint_target=q_ref+normalized_action*scale"
    assert manifest["deployment_controller"]["outputs"] == [
        "normalized_action",
        "joint_target",
    ]
