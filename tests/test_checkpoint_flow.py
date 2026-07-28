"""Standard RSL-RL checkpoint handoff tests for the S1-to-S2 transition."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from g1_rickshaw_lab.checkpoint_handoff import initialize_student_runner


def _runner() -> SimpleNamespace:
    return SimpleNamespace(
        alg=SimpleNamespace(
            _raw_actor=nn.Linear(3, 2),
            _raw_critic=nn.Linear(4, 1),
        )
    )


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
    runner = _runner()

    initialize_student_runner(runner, teacher=teacher, context=context)

    for actual, expected in zip(
        runner.alg._raw_actor.parameters(), source_actor.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        runner.alg._raw_critic.parameters(), source_critic.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_fresh_s2_uses_strict_state_loading(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.pt"
    context = tmp_path / "context.pt"
    torch.save({"critic_state_dict": nn.Linear(4, 1).state_dict()}, teacher)
    torch.save({"student_state_dict": nn.Linear(5, 2).state_dict()}, context)

    with pytest.raises(RuntimeError):
        initialize_student_runner(_runner(), teacher=teacher, context=context)


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
        initialize_student_runner(_runner(), teacher=teacher, context=context)
