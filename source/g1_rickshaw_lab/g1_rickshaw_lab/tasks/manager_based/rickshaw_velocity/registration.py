"""Register the G1 rickshaw tasks with Mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from . import (
    DISTILLATION_TASK_ID,
    STUDENT_TASK_ID,
    TRAIN_TASK_ID,
    g1_rickshaw_env_cfg,
)
from .agents.rsl_rl_cfg import (
    g1_rickshaw_distillation_runner_cfg,
    g1_rickshaw_student_ppo_runner_cfg,
    g1_rickshaw_teacher_ppo_runner_cfg,
)
from .agents.runners import (
    RickshawDistillationRunner,
    RickshawStudentRunner,
    RickshawTeacherRunner,
)

for task_id, reward_curriculum, runner_cfg, runner_cls in (
    (
        TRAIN_TASK_ID,
        "s0",
        g1_rickshaw_teacher_ppo_runner_cfg(),
        RickshawTeacherRunner,
    ),
    (
        DISTILLATION_TASK_ID,
        None,
        g1_rickshaw_distillation_runner_cfg(),
        RickshawDistillationRunner,
    ),
    (
        STUDENT_TASK_ID,
        "s2",
        g1_rickshaw_student_ppo_runner_cfg(),
        RickshawStudentRunner,
    ),
):
    register_mjlab_task(
        task_id=task_id,
        env_cfg=g1_rickshaw_env_cfg(
            play=False,
            reward_curriculum=reward_curriculum,
        ),
        play_env_cfg=g1_rickshaw_env_cfg(
            play=True,
            reward_curriculum=None,
        ),
        rl_cfg=runner_cfg,
        runner_cls=runner_cls,
    )
