"""Pure-PyTorch context models for the G1 rickshaw task."""

from g1_rickshaw_lab.policy_schema import CRITIC_PRIVILEGED_DIM as CRITIC_PRIVILEGE_DIM

from .context_encoder import CausalBlock, ContextEncoder, temporal_receptive_field
from .teacher_model import (
    DYNAMIC_PRIVILEGE_DIM,
    STATIC_PRIVILEGE_DIM,
    TeacherEncoder,
)

__all__ = [
    "CausalBlock",
    "ContextEncoder",
    "CRITIC_PRIVILEGE_DIM",
    "DYNAMIC_PRIVILEGE_DIM",
    "STATIC_PRIVILEGE_DIM",
    "TeacherEncoder",
    "temporal_receptive_field",
]
