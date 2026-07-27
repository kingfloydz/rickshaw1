"""Online teacher-to-student distillation following RSL-RL 5.4.0."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from g1_rickshaw_lab.policy_schema import ACTION_DIM, ACTOR_OBSERVATION_DIM

from .actor_critic import G1RickshawStudentActor


class OnlineDistillationStorage:
    """Store only the observations and teacher targets used by S1 updates."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        history_length: int,
        device: torch.device,
    ) -> None:
        self.num_steps = num_steps
        self.current = torch.empty(
            num_steps, num_envs, ACTOR_OBSERVATION_DIM, device=device
        )
        self.history = torch.empty(
            num_steps,
            num_envs,
            history_length,
            ACTOR_OBSERVATION_DIM,
            device=device,
        )
        self.teacher_actions = torch.empty(
            num_steps, num_envs, ACTION_DIM, device=device
        )
        self.step = 0

    def add(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps:
            raise RuntimeError("distillation rollout storage is full")
        self.current[self.step].copy_(current)
        self.history[self.step].copy_(history)
        self.teacher_actions[self.step].copy_(teacher_actions)
        self.step += 1

    def batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.step != self.num_steps:
            raise RuntimeError("distillation rollout storage is incomplete")
        for step in range(self.num_steps):
            yield self.current[step], self.history[step], self.teacher_actions[step]

    def clear(self) -> None:
        self.step = 0


class OnlineStudentDistillation:
    """Collect with the student and regress its mean action to the teacher."""

    def __init__(
        self,
        student: G1RickshawStudentActor,
        teacher: Any,
        storage: OnlineDistillationStorage,
        *,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.storage = storage
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
        self.student.train()
        self.teacher.eval()

    def act(self, observation: Any) -> torch.Tensor:
        """Sample the student action and record the teacher's deterministic target."""

        current = observation["policy"]
        history = observation["history"]
        with torch.no_grad():
            student_actions = self.student(current, history).sample()
            teacher_actions = self.teacher(observation)
            self.storage.add(current, history, teacher_actions)
        return student_actions

    def update(self) -> dict[str, float]:
        """Apply one optimizer step to the complete online rollout."""

        self.optimizer.zero_grad(set_to_none=True)
        behavior = torch.zeros((), device=self.storage.current.device)
        for current, history, teacher_actions in self.storage.batches():
            student_actions = self.student(current, history).mean
            step_loss = F.mse_loss(student_actions, teacher_actions)
            (step_loss / self.storage.num_steps).backward()
            behavior += step_loss.detach() / self.storage.num_steps
        gradient_norm = nn.utils.clip_grad_norm_(
            self.student.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.storage.clear()
        return {
            "behavior": float(behavior),
            "gradient_norm": float(gradient_norm),
        }

    def process_env_step(self, observation: Any) -> None:
        """Update student normalization from the post-step observation."""

        self.student.obs_normalizer.update(observation["policy"])


__all__ = ["OnlineDistillationStorage", "OnlineStudentDistillation"]
