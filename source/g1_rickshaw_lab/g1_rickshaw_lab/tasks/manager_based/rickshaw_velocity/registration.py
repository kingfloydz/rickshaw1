"""Register the G1 rickshaw tasks with Mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from . import (
    DISTILLATION_TASK_ID,
    HISTORY_91_DISTILLATION_TASK_ID,
    HISTORY_91_STUDENT_TASK_ID,
    HISTORY_91_TEACHER_TASK_ID,
    STUDENT_TASK_ID,
    TRAIN_TASK_ID,
    g1_rickshaw_env_cfg,
)
from .agents.distillation_runner import MjlabDistillationRunner
from .agents.rsl_rl_cfg import (
    g1_rickshaw_distillation_runner_cfg,
    g1_rickshaw_student_ppo_runner_cfg,
    g1_rickshaw_teacher_ppo_runner_cfg,
)

for task_id, kind, history_length in (
    (TRAIN_TASK_ID, "teacher", 61),
    (DISTILLATION_TASK_ID, "distillation", 61),
    (STUDENT_TASK_ID, "student", 61),
    (HISTORY_91_TEACHER_TASK_ID, "teacher", 91),
    (HISTORY_91_DISTILLATION_TASK_ID, "distillation", 91),
    (HISTORY_91_STUDENT_TASK_ID, "student", 91),
):
    if kind == "distillation":
        runner_cfg = g1_rickshaw_distillation_runner_cfg(history_length=history_length)
        runner_cls = MjlabDistillationRunner
    elif kind == "student":
        runner_cfg = g1_rickshaw_student_ppo_runner_cfg(history_length=history_length)
        runner_cls = None
    else:
        runner_cfg = g1_rickshaw_teacher_ppo_runner_cfg(history_length=history_length)
        runner_cls = None
    register_mjlab_task(
        task_id=task_id,
        env_cfg=g1_rickshaw_env_cfg(play=False, history_length=history_length),
        play_env_cfg=g1_rickshaw_env_cfg(play=True, history_length=history_length),
        rl_cfg=runner_cfg,
        runner_cls=runner_cls,
    )
