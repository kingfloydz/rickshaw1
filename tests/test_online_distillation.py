from __future__ import annotations

import torch

from g1_rickshaw_lab.policy_schema import ACTION_DIM, ACTOR_OBSERVATION_DIM
from g1_rickshaw_lab.rl import (
    G1RickshawStudentActor,
    OnlineDistillationStorage,
    OnlineStudentDistillation,
)


class _Teacher:
    def eval(self):
        return self

    def __call__(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return observation["teacher_action"]


def _observation(num_envs: int) -> dict[str, torch.Tensor]:
    return {
        "policy": torch.randn(num_envs, ACTOR_OBSERVATION_DIM),
        "history": torch.randn(num_envs, 61, ACTOR_OBSERVATION_DIM),
        "teacher_action": torch.ones(num_envs, ACTION_DIM),
    }


def test_student_actions_drive_collection_and_teacher_actions_are_targets() -> None:
    torch.manual_seed(3)
    student = G1RickshawStudentActor()
    storage = OnlineDistillationStorage(2, 2, 61, torch.device("cpu"))
    algorithm = OnlineStudentDistillation(student, _Teacher(), storage)
    observation = _observation(2)

    action = algorithm.act(observation)
    algorithm.process_env_step(observation)

    assert action.shape == (2, ACTION_DIM)
    assert not torch.equal(action, observation["teacher_action"])
    torch.testing.assert_close(storage.teacher_actions[0], observation["teacher_action"])
    assert student.obs_normalizer.count.item() == 2


def test_online_update_consumes_one_complete_rollout() -> None:
    torch.manual_seed(4)
    student = G1RickshawStudentActor()
    storage = OnlineDistillationStorage(2, 2, 61, torch.device("cpu"))
    algorithm = OnlineStudentDistillation(student, _Teacher(), storage)
    observation = _observation(2)
    for _ in range(2):
        algorithm.act(observation)
    before = student.actor.network[-1].bias.detach().clone()

    metrics = algorithm.update()

    assert set(metrics) == {"behavior", "gradient_norm"}
    assert metrics["behavior"] > 0.0
    assert storage.step == 0
    assert not torch.equal(student.actor.network[-1].bias, before)
