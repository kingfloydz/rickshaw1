"""Standard RSL-RL checkpoint handoff tests for the S1-to-S2 transition."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from torch import nn

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents.runners import (
    initialize_student_models,
)


class _Algorithm:
    def __init__(self) -> None:
        self._raw_actor = nn.Linear(3, 2)
        self._raw_critic = nn.Linear(4, 1)

    def load(self, checkpoint: dict, load_cfg: dict, strict: bool) -> bool:
        assert load_cfg == {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        }
        self._raw_actor.load_state_dict(checkpoint["actor_state_dict"], strict=strict)
        self._raw_critic.load_state_dict(checkpoint["critic_state_dict"], strict=strict)
        return False


def test_fresh_s2_loads_only_upstream_actor_and_critic_states(
    tmp_path: Path,
) -> None:
    source_actor = nn.Linear(3, 2)
    source_critic = nn.Linear(4, 1)
    teacher = tmp_path / "teacher.pt"
    context = tmp_path / "context.pt"
    torch.save(
        {
            "actor_state_dict": nn.Linear(3, 2).state_dict(),
            "critic_state_dict": source_critic.state_dict(),
            "optimizer_state_dict": {},
            "iter": 10,
            "infos": {"env_state": {"common_step_counter": 100}},
        },
        teacher,
    )
    torch.save(
        {
            "student_state_dict": source_actor.state_dict(),
            "teacher_state_dict": nn.Linear(3, 2).state_dict(),
            "optimizer_state_dict": {},
            "iter": 20,
            "infos": {"env_state": {"common_step_counter": 200}},
        },
        context,
    )
    algorithm = _Algorithm()

    initialize_student_models(algorithm, str(teacher), str(context))

    for actual, expected in zip(
        algorithm._raw_actor.parameters(), source_actor.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        algorithm._raw_critic.parameters(), source_critic.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_fresh_s2_uses_strict_state_loading(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.pt"
    context = tmp_path / "context.pt"
    torch.save({"critic_state_dict": nn.Linear(4, 1).state_dict()}, teacher)
    torch.save({"student_state_dict": nn.Linear(5, 2).state_dict()}, context)

    with pytest.raises(RuntimeError):
        initialize_student_models(_Algorithm(), str(teacher), str(context))


@pytest.mark.parametrize(
    ("teacher_payload", "context_payload", "message"),
    (
        ({}, {"student_state_dict": {}}, "critic_state_dict"),
        ({"critic_state_dict": {}}, {}, "student_state_dict"),
    ),
)
def test_fresh_s2_requires_standard_checkpoint_keys(
    tmp_path: Path,
    teacher_payload: dict,
    context_payload: dict,
    message: str,
) -> None:
    teacher = tmp_path / "teacher.pt"
    context = tmp_path / "context.pt"
    torch.save(teacher_payload, teacher)
    torch.save(context_payload, context)

    with pytest.raises(KeyError, match=message):
        initialize_student_models(_Algorithm(), str(teacher), str(context))
