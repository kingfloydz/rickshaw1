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

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        loading_teacher = "actor_state_dict" in checkpoint
        infos = super().load(path, load_cfg, strict, map_location)
        if loading_teacher:
            student = self.alg._raw_student
            teacher = self.alg._raw_teacher
            student.policy.load_state_dict(teacher.policy.state_dict(), strict=True)
            student.policy_obs_normalizer.load_state_dict(teacher.policy_obs_normalizer.state_dict(), strict=True)
        return infos


def initialize_student_models(algorithm, teacher_checkpoint: str, context_checkpoint: str) -> None:
    teacher = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    context = torch.load(context_checkpoint, map_location="cpu", weights_only=False)
    critic_state = teacher["critic_state_dict"]
    student_state = context["student_state_dict"]
    algorithm._raw_actor.load_state_dict(student_state, strict=True)
    algorithm._raw_critic.load_state_dict(critic_state, strict=True)


class RickshawStudentRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        checkpoint = train_cfg.pop("checkpoint_file", None)
        teacher_checkpoint = train_cfg.pop("teacher_checkpoint", None)
        context_checkpoint = train_cfg.pop("context_checkpoint", None)
        super().__init__(env, train_cfg, log_dir, device)
        if checkpoint is not None:
            self.load(checkpoint, map_location=device)
        elif teacher_checkpoint is not None and context_checkpoint is not None:
            initialize_student_models(self.alg, teacher_checkpoint, context_checkpoint)


__all__ = [
    "RickshawDistillationRunner",
    "RickshawStudentRunner",
    "RickshawTeacherRunner",
]
