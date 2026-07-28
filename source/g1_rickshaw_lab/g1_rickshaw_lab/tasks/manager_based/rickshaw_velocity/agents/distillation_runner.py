"""Mjlab environment-state support for the official RSL-RL distillation runner."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.runners import DistillationRunner


class MjlabDistillationRunner(MjlabOnPolicyRunner, DistillationRunner):
    """Use upstream distillation while retaining Mjlab checkpointed env state."""

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint root must be a mapping")
        loading_teacher = "actor_state_dict" in checkpoint
        infos = super().load(path, load_cfg, strict, map_location)
        if loading_teacher:
            student = self.alg._raw_student
            teacher = self.alg._raw_teacher
            student.policy.load_state_dict(teacher.policy.state_dict(), strict=True)
            student.policy_obs_normalizer.load_state_dict(teacher.policy_obs_normalizer.state_dict(), strict=True)
        return infos


__all__ = ["MjlabDistillationRunner"]
