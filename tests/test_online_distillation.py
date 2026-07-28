from __future__ import annotations

import inspect
from collections import OrderedDict
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("rsl_rl")

from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from g1_rickshaw_lab.policy_schema import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_DIM,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
)
from g1_rickshaw_lab.rl.rsl_rl_models import (
    RslRickshawActorModel,
    RslRickshawCriticModel,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents.rsl_rl_cfg import (
    g1_rickshaw_distillation_runner_cfg,
)


def _observation(num_envs: int) -> TensorDict:
    return TensorDict(
        {
            "policy_sequence": torch.randn(num_envs, 62, ACTOR_OBSERVATION_DIM),
            "teacher_dynamic_sequence": torch.randn(num_envs, 62, TEACHER_DYNAMIC_DIM),
            "teacher_static": torch.randn(num_envs, TEACHER_STATIC_DIM),
            "critic_policy": torch.randn(num_envs, ACTOR_OBSERVATION_DIM),
            "critic": torch.randn(num_envs, CRITIC_PRIVILEGED_DIM),
        },
        batch_size=[num_envs],
    )


def _algorithm(num_steps: int = 3) -> tuple[Distillation, TensorDict]:
    observation = _observation(2)
    student = RslRickshawActorModel(
        observation,
        {"student": ["policy_sequence"]},
        "student",
        ACTION_DIM,
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    teacher = RslRickshawActorModel(
        observation,
        {
            "teacher": [
                "policy_sequence",
                "teacher_dynamic_sequence",
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


def test_actor_reuses_upstream_mlp_and_gaussian_distribution() -> None:
    model = RslRickshawActorModel(
        _observation(2),
        {"student": ["policy_sequence"]},
        "student",
        ACTION_DIM,
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.7,
            "std_type": "log",
        },
    )

    assert isinstance(model, MLPModel)
    assert isinstance(model.distribution, GaussianDistribution)
    assert model.distribution.std_type == "log"
    assert model.mlp[0].in_features == ACTOR_OBSERVATION_DIM + model.latent_dim

    observation = _observation(2)
    scripted = torch.jit.script(model.eval().as_jit())
    with torch.inference_mode():
        expected = model(observation)
        actual = scripted(
            observation["policy_sequence"][:, -1],
            observation["policy_sequence"][:, :-1],
        )
    torch.testing.assert_close(actual, expected)


def test_previous_actor_adapter_checkpoint_is_migrated_strictly() -> None:
    source = RslRickshawActorModel(
        _observation(2),
        {"student": ["policy_sequence"]},
        "student",
        ACTION_DIM,
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    legacy = OrderedDict()
    for key, value in source.state_dict().items():
        if key.startswith("mlp."):
            key = "policy.network." + key.removeprefix("mlp.")
        elif key == "distribution.std_param":
            key = "policy.std_param"
        elif key.startswith("obs_normalizer."):
            key = "policy_obs_normalizer." + key.removeprefix("obs_normalizer.")
        legacy[key] = value.clone()

    restored = RslRickshawActorModel(
        _observation(2),
        {"student": ["policy_sequence"]},
        "student",
        ACTION_DIM,
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    restored.load_state_dict(legacy, strict=True)

    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_previous_critic_adapter_checkpoint_merges_split_normalizers() -> None:
    observation = _observation(2)
    groups = {"critic": ["critic_policy", "critic"]}
    source = RslRickshawCriticModel(
        observation,
        groups,
        "critic",
        1,
        hidden_dims=(512, 256, 128),
        obs_normalization=True,
    )
    legacy = OrderedDict()
    for key, value in source.state_dict().items():
        if key.startswith("mlp."):
            legacy["value.network." + key.removeprefix("mlp.")] = value.clone()
        elif key.startswith("obs_normalizer."):
            suffix = key.removeprefix("obs_normalizer.")
            if suffix == "count":
                legacy[f"policy_obs_normalizer.{suffix}"] = value.clone()
                legacy[f"privileged_obs_normalizer.{suffix}"] = value.clone()
            else:
                legacy[f"policy_obs_normalizer.{suffix}"] = value[
                    ..., :ACTOR_OBSERVATION_DIM
                ].clone()
                legacy[f"privileged_obs_normalizer.{suffix}"] = value[
                    ..., ACTOR_OBSERVATION_DIM:
                ].clone()

    restored = RslRickshawCriticModel(
        observation,
        groups,
        "critic",
        1,
        hidden_dims=(512, 256, 128),
        obs_normalization=True,
    )
    restored.load_state_dict(legacy, strict=True)

    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


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
    assert algorithm.student.obs_normalizer.count.item() == 2


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
    before = algorithm.student.mlp[-1].bias.detach().clone()

    metrics = algorithm.update()

    assert set(metrics) == {"behavior"}
    assert metrics["behavior"] > 0.0
    assert algorithm.storage.step == 0
    assert not torch.equal(algorithm.student.mlp[-1].bias, before)
