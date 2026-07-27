"""G1 rickshaw velocity task configuration."""

from .env_cfg import g1_rickshaw_env_cfg

TRAIN_TASK_ID = "Mjlab-G1-Rickshaw-Slopes-Teacher"
STUDENT_TASK_ID = "Mjlab-G1-Rickshaw-Slopes-Student"
HISTORY_91_TEACHER_TASK_ID = TRAIN_TASK_ID + "-H91"
HISTORY_91_STUDENT_TASK_ID = STUDENT_TASK_ID + "-H91"

__all__ = [
    "HISTORY_91_STUDENT_TASK_ID",
    "HISTORY_91_TEACHER_TASK_ID",
    "STUDENT_TASK_ID",
    "TRAIN_TASK_ID",
    "g1_rickshaw_env_cfg",
]
