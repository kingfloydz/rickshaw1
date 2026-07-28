from __future__ import annotations

import inspect
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("rsl_rl")

from rsl_rl.algorithms import Distillation
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from g1_rickshaw_lab.policy_schema import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
)
from g1_rickshaw_lab.rl.rsl_rl_models import RslRickshawActorModel
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents.rsl_rl_cfg import (
    g1_rickshaw_distillation_runner_cfg,
)


def _observation(num_envs: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(num_envs, ACTOR_OBSERVATION_DIM),
            "history": torch.randn(num_envs, 61, ACTOR_OBSERVATION_DIM),
            "teacher_dynamic_history": torch.randn(num_envs, 61, TEACHER_DYNAMIC_DIM),
            "teacher_static": torch.randn(num_envs, TEACHER_STATIC_DIM),
        },
        batch_size=[num_envs],
    )


def _algorithm(num_steps: int = 3) -> tuple[Distillation, TensorDict]:
    observation = _observation(2)
    student = RslRickshawActorModel(
        observation,
        {"student": ["policy", "history"]},
        "student",
        ACTION_DIM,
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    teacher = RslRickshawActorModel(
        observation,
        {
            "teacher": [
                "policy",
                "history",
                "teacher_dynamic_history",
                "teacher_static",
            ]
        },
        "teacher",
        ACTION_DIM,
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    storage = RolloutStorage("distillation", 2, num_steps, observation, [ACTION_DIM])
    algorithm = Distillation(
        student,
        teacher,
        storage,
        gradient_length=num_steps,
    )
    return algorithm, observation


def test_project_uses_official_v5_4_distillation_defaults() -> None:
    parameters = inspect.signature(Distillation.__init__).parameters
    assert Distillation.__module__ == "rsl_rl.algorithms.distillation"
    assert parameters["num_learning_epochs"].default == 1
    assert parameters["gradient_length"].default == 15
    assert parameters["learning_rate"].default == 1.0e-3
    assert parameters["max_grad_norm"].default is None


def test_upstream_distillation_constructs_from_registered_config() -> None:
    cfg = asdict(g1_rickshaw_distillation_runner_cfg())
    cfg["multi_gpu"] = None

    algorithm = Distillation.construct_algorithm(
        _observation(2),
        SimpleNamespace(num_actions=ACTION_DIM, num_envs=2),
        cfg,
        "cpu",
    )

    assert isinstance(algorithm, Distillation)


def test_student_actions_drive_collection_and_teacher_actions_are_targets() -> None:
    torch.manual_seed(3)
    algorithm, observation = _algorithm()

    with torch.inference_mode():
        teacher_action = algorithm.teacher(observation)
        action = algorithm.act(observation)
        algorithm.process_env_step(
            observation,
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            {},
        )

    assert action.shape == (2, ACTION_DIM)
    assert not torch.equal(action, teacher_action)
    torch.testing.assert_close(algorithm.storage.privileged_actions[0], teacher_action)
    assert algorithm.student.policy_obs_normalizer.count.item() == 2


def test_online_update_consumes_one_complete_rollout() -> None:
    torch.manual_seed(4)
    algorithm, observation = _algorithm()
    with torch.inference_mode():
        for _ in range(3):
            algorithm.act(observation)
            algorithm.process_env_step(
                observation,
                torch.zeros(2),
                torch.zeros(2, dtype=torch.bool),
                {},
            )
    before = algorithm.student.policy.network[-1].bias.detach().clone()

    metrics = algorithm.update()

    assert set(metrics) == {"behavior"}
    assert metrics["behavior"] > 0.0
    assert algorithm.storage.step == 0
    assert not torch.equal(algorithm.student.policy.network[-1].bias, before)
