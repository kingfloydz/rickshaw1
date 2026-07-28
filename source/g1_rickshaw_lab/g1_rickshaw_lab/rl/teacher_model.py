"""Temporal privileged encoder used only by the S0 teacher."""

from __future__ import annotations

import torch
from torch import nn

from g1_rickshaw_lab.policy_schema import (
    DEFAULT_CONTEXT_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
    TEACHER_STATIC_FEATURE_DIM,
    validate_context_dim,
    validate_history_length,
)

from .context_encoder import (
    DILATIONS,
    FEATURE_DIM,
    HISTORY_KERNEL_SIZES,
    OBSERVATION_DIM,
    CausalBlock,
)

DYNAMIC_PRIVILEGE_DIM = TEACHER_DYNAMIC_DIM
STATIC_PRIVILEGE_DIM = TEACHER_STATIC_DIM


class TeacherEncoder(nn.Module):
    """Fuse observation/physical histories with episode-static physics."""

    def __init__(
        self,
        latent_dim: int = DEFAULT_CONTEXT_DIM,
        history_length: int = HISTORY_LENGTH,
    ) -> None:
        super().__init__()
        self.latent_dim = validate_context_dim(latent_dim)
        self.history_length = validate_history_length(history_length)
        self.kernel_size = HISTORY_KERNEL_SIZES[self.history_length]
        self.temporal_input = nn.Conv1d(
            OBSERVATION_DIM + DYNAMIC_PRIVILEGE_DIM,
            FEATURE_DIM,
            kernel_size=1,
        )
        self.blocks = nn.Sequential(*(CausalBlock(FEATURE_DIM, dilation, self.kernel_size) for dilation in DILATIONS))
        self.static = nn.Sequential(
            nn.Linear(STATIC_PRIVILEGE_DIM, TEACHER_STATIC_FEATURE_DIM),
            nn.ELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(FEATURE_DIM + TEACHER_STATIC_FEATURE_DIM, FEATURE_DIM),
            nn.ELU(),
            nn.Linear(FEATURE_DIM, self.latent_dim),
        )

    def forward(
        self,
        observation_history: torch.Tensor,
        dynamic_privilege_history: torch.Tensor,
        static_privilege: torch.Tensor,
    ) -> torch.Tensor:
        temporal_history = torch.cat((observation_history, dynamic_privilege_history), dim=-1)
        temporal = self.blocks(self.temporal_input(temporal_history.transpose(1, 2)))[:, :, -1]
        static = self.static(static_privilege)
        return self.context(torch.cat((temporal, static), dim=-1))
