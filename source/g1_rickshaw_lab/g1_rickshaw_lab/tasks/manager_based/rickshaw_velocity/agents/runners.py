"""Task runners extending Mjlab at its registered runner boundary."""

from __future__ import annotations

import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.runners import DistillationRunner


class RickshawTeacherRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        checkpoint = train_cfg.pop("checkpoint_file", None)
        super().__init__(env, train_cfg, log_dir, device)
        if checkpoint is not None:
            self.load(checkpoint, map_location=device)


class RickshawDistillationRunner(MjlabOnPolicyRunner, DistillationRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        teacher_checkpoint = train_cfg.pop("teacher_checkpoint", None)
        super().__init__(env, train_cfg, log_dir, device)
        if teacher_checkpoint is not None:
            self.load(teacher_checkpoint, map_location=device)
            student = self.alg._raw_student
            teacher = self.alg._raw_teacher
            student.mlp.load_state_dict(teacher.mlp.state_dict(), strict=True)
            student.distribution.load_state_dict(teacher.distribution.state_dict(), strict=True)
            student.obs_normalizer.load_state_dict(teacher.obs_normalizer.state_dict(), strict=True)


def initialize_student_models(algorithm, teacher_checkpoint: str, context_checkpoint: str) -> None:
    teacher = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    context = torch.load(context_checkpoint, map_location="cpu", weights_only=False)
    algorithm.load(
        {
            "actor_state_dict": context["student_state_dict"],
            "critic_state_dict": teacher["critic_state_dict"],
        },
        {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
        strict=True,
    )


class RickshawStudentRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        checkpoint = train_cfg.pop("checkpoint_file", None)
        teacher_checkpoint = train_cfg.pop("teacher_checkpoint", None)
        context_checkpoint = train_cfg.pop("context_checkpoint", None)
        if (teacher_checkpoint is None) != (context_checkpoint is None):
            raise ValueError("teacher_checkpoint and context_checkpoint must be provided together")
        super().__init__(env, train_cfg, log_dir, device)
        if checkpoint is not None:
            self.load(checkpoint, map_location=device)
        elif teacher_checkpoint is not None and context_checkpoint is not None:
            initialize_student_models(self.alg, teacher_checkpoint, context_checkpoint)
