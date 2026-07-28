"""Minimal state-dict handoff from upstream S0/S1 checkpoints to S2 PPO."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _load_checkpoint(path: Path, label: str) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{label} checkpoint root must be a mapping")
    return checkpoint


def initialize_student_runner(runner: Any, *, teacher: Path, context: Path) -> None:
    """Load the S1 student and S0 critic into a fresh upstream PPO runner."""

    teacher_checkpoint = _load_checkpoint(teacher, "teacher")
    context_checkpoint = _load_checkpoint(context, "distillation")
    actor_state = context_checkpoint.get("student_state_dict")
    critic_state = teacher_checkpoint.get("critic_state_dict")
    if not isinstance(actor_state, Mapping):
        raise KeyError("distillation checkpoint is missing student_state_dict")
    if not isinstance(critic_state, Mapping):
        raise KeyError("teacher checkpoint is missing critic_state_dict")
    runner.alg._raw_actor.load_state_dict(actor_state, strict=True)
    runner.alg._raw_critic.load_state_dict(critic_state, strict=True)


__all__ = ["initialize_student_runner"]
