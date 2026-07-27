from __future__ import annotations

import pytest
import torch

from g1_rickshaw_lab.policy_schema import ACTOR_OBSERVATION_DIM
from g1_rickshaw_lab.training_contract import (
    GUIDE_MAX_ITERATIONS,
    GUIDE_TRAINING_PARAMETERS,
    extract_policy_observation_normalizer_state,
    extract_student_rsl_actor_state,
    guide_max_iterations,
    rollout_scaled_iterations,
    s0_remaining_learning_iterations,
    s2_remaining_learning_iterations,
    training_artifact_interval,
    validate_student_checkpoint_architecture,
    validate_teacher_checkpoint_architecture,
)


def test_observation_normalizer_state_survives_s0_s1_s2_handoff() -> None:
    normalizer = {
        "_mean": torch.randn(1, ACTOR_OBSERVATION_DIM),
        "_var": torch.rand(1, ACTOR_OBSERVATION_DIM),
        "_std": torch.rand(1, ACTOR_OBSERVATION_DIM),
        "count": torch.tensor(1024),
    }
    teacher = {
        "actor_state_dict": {
            **{f"policy_obs_normalizer.{key}": value for key, value in normalizer.items()},
        }
    }
    extracted = extract_policy_observation_normalizer_state(teacher)
    assert set(extracted) == set(normalizer)

    student = {
        "model_state_dict": {
            "context_encoder.input.weight": torch.zeros(64, ACTOR_OBSERVATION_DIM, 1),
            "actor.network.0.weight": torch.zeros(512, ACTOR_OBSERVATION_DIM + 16),
            **{f"obs_normalizer.{key}": value for key, value in normalizer.items()},
        }
    }
    bootstrap = extract_student_rsl_actor_state(student)
    for key, value in normalizer.items():
        torch.testing.assert_close(bootstrap[f"policy_obs_normalizer.{key}"], value)


def test_mainline_has_fixed_stage_budgets_and_sloped_terrain() -> None:
    assert GUIDE_MAX_ITERATIONS == {
        "s0_teacher": 30000,
        "s1_context_distillation": 2000,
        "s2_student_ppo": 30000,
    }
    assert guide_max_iterations("s0_teacher") == 30000
    assert GUIDE_TRAINING_PARAMETERS["s0_teacher"] == {
        "domain_randomization": "startup_fixed",
        "terrain": "nineteen_slopes",
        "observation_noise": "unitree_g1_uniform",
    }
    with pytest.raises(ValueError, match="unknown training stage"):
        guide_max_iterations("legacy")


def test_s1_uses_online_rsl_distillation() -> None:
    assert GUIDE_TRAINING_PARAMETERS["s1_context_distillation"] == {
        "implementation": "rsl_rl_v5.4.0_online_distillation",
        "learning_rate": 1.0e-3,
        "loss_type": "mse",
        "num_learning_epochs": 1,
        "gradient_clip": 1.0,
        "actor_initialized_from_teacher": True,
        "student_actions_drive_environment": True,
        "teacher_target": "deterministic_action_mean",
        "save_interval": 50,
        "deterministic_algorithms": False,
    }


@pytest.mark.parametrize(
    ("rollout_steps", "s0_iterations", "s2_iterations", "artifact_interval"),
    (
        (24, 30000, 30000, 50),
        (48, 15000, 15000, 50),
        (64, 11250, 11250, 50),
    ),
)
def test_rollout_variants_preserve_transition_and_artifact_budgets(
    rollout_steps: int,
    s0_iterations: int,
    s2_iterations: int,
    artifact_interval: int,
) -> None:
    assert guide_max_iterations("s0_teacher", rollout_steps) == s0_iterations
    assert guide_max_iterations("s2_student_ppo", rollout_steps) == s2_iterations
    assert rollout_scaled_iterations(30000, rollout_steps) == s0_iterations
    assert training_artifact_interval(rollout_steps) == artifact_interval
    assert s0_iterations * rollout_steps == 30000 * 24
    assert s2_iterations * rollout_steps == 30000 * 24
    assert artifact_interval == 50


@pytest.mark.parametrize(
    ("remaining", "function"),
    (
        (400, s0_remaining_learning_iterations),
        (400, s2_remaining_learning_iterations),
    ),
)
def test_ppo_resume_uses_only_the_remaining_iteration_budget(
    remaining: int,
    function,
) -> None:
    assert function(requested_iterations=2000, completed_iterations=1600) == remaining
    with pytest.raises(ValueError, match="exceeds"):
        function(requested_iterations=2000, completed_iterations=2001)


@pytest.mark.parametrize("latent_dim", (8, 16, 24, 32))
def test_checkpoint_tensor_widths_match_the_recorded_latent(latent_dim: int) -> None:
    configuration = {"training_parameters": {"latent_dim": latent_dim}}
    student = {
        "model_state_dict": {
            "context_encoder.context.weight": torch.zeros(latent_dim, 64),
            "actor.network.0.weight": torch.zeros(512, ACTOR_OBSERVATION_DIM + latent_dim),
        }
    }
    teacher = {
        "actor_state_dict": {
            "encoder.context.weight": torch.zeros(latent_dim, ACTOR_OBSERVATION_DIM),
            "policy.network.0.weight": torch.zeros(512, ACTOR_OBSERVATION_DIM + latent_dim),
        }
    }
    validate_student_checkpoint_architecture(student, configuration)
    validate_teacher_checkpoint_architecture(teacher, configuration)

    student["model_state_dict"]["actor.network.0.weight"] = torch.zeros(
        512, ACTOR_OBSERVATION_DIM + 16
    )
    if latent_dim == 16:
        student["model_state_dict"]["actor.network.0.weight"] = torch.zeros(
            512, ACTOR_OBSERVATION_DIM + 17
        )
    with pytest.raises(ValueError, match="recorded latent width"):
        validate_student_checkpoint_architecture(student, configuration)
