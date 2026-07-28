"""G1 rickshaw velocity task configuration."""

from .env_cfg import g1_rickshaw_env_cfg

TRAIN_TASK_ID = "Mjlab-G1-Rickshaw-Slopes-Teacher"
DISTILLATION_TASK_ID = "Mjlab-G1-Rickshaw-Slopes-Distillation"
STUDENT_TASK_ID = "Mjlab-G1-Rickshaw-Slopes-Student"

__all__ = [
    "DISTILLATION_TASK_ID",
    "STUDENT_TASK_ID",
    "TRAIN_TASK_ID",
    "g1_rickshaw_env_cfg",
]
