"""RSL-RL configurations for teacher, distillation, and student training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from mjlab.rl import (
    RslRlBaseRunnerCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from g1_rickshaw_lab.policy_schema import DEFAULT_CONTEXT_DIM, HISTORY_LENGTH

DEFAULT_ROLLOUT_STEPS = 24
DEFAULT_TEACHER_ITERATIONS = 6_000
DEFAULT_DISTILLATION_ITERATIONS = 6_000
DEFAULT_STUDENT_ITERATIONS = 800
DEFAULT_SAVE_INTERVAL = 50


@dataclass
class RickshawActorCfg(RslRlModelCfg):
    latent_dim: int = DEFAULT_CONTEXT_DIM
    history_length: int = HISTORY_LENGTH


@dataclass
class TeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    checkpoint_file: str | None = None


@dataclass
class StudentRunnerCfg(RslRlOnPolicyRunnerCfg):
    checkpoint_file: str | None = None
    teacher_checkpoint: str | None = None
    context_checkpoint: str | None = None


@dataclass
class DistillationAlgorithmCfg:
    class_name: str = "Distillation"
    num_learning_epochs: int = 1
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float | None = None
    loss_type: str = "mse"
    optimizer: str = "adam"


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
    student: RickshawActorCfg = field(default_factory=RickshawActorCfg)
    teacher: RickshawActorCfg = field(default_factory=RickshawActorCfg)
    algorithm: DistillationAlgorithmCfg = field(default_factory=DistillationAlgorithmCfg)
    teacher_checkpoint: str | None = None


def _actor_cfg(base: RslRlModelCfg, *, latent_dim: int, history_length: int) -> RickshawActorCfg:
    values = asdict(base)
    values["class_name"] = "g1_rickshaw_lab.rl.rsl_rl_models:RslRickshawActorModel"
    return RickshawActorCfg(
        **values,
        latent_dim=latent_dim,
        history_length=history_length,
    )


def _g1_training_components(
    *, latent_dim: int, history_length: int
) -> tuple[RickshawActorCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg]:
    base = unitree_g1_ppo_runner_cfg()
    return (
        _actor_cfg(base.actor, latent_dim=latent_dim, history_length=history_length),
        base.critic,
        base.algorithm,
    )


def g1_rickshaw_teacher_ppo_runner_cfg(
    *,
    latent_dim: int = DEFAULT_CONTEXT_DIM,
    history_length: int = HISTORY_LENGTH,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
) -> TeacherRunnerCfg:
    actor, critic, algorithm = _g1_training_components(latent_dim=latent_dim, history_length=history_length)
    return TeacherRunnerCfg(
        num_steps_per_env=rollout_steps,
        max_iterations=DEFAULT_TEACHER_ITERATIONS,
        save_interval=DEFAULT_SAVE_INTERVAL,
        experiment_name="g1_rickshaw_teacher",
        run_name="s0",
        logger="tensorboard",
        obs_groups={
            "actor": ("policy_sequence", "teacher_dynamic_sequence", "teacher_static"),
            "critic": ("critic_policy", "critic"),
        },
        actor=actor,
        critic=critic,
        algorithm=algorithm,
    )


def g1_rickshaw_student_ppo_runner_cfg(
    *,
    latent_dim: int = DEFAULT_CONTEXT_DIM,
    history_length: int = HISTORY_LENGTH,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
) -> StudentRunnerCfg:
    actor, critic, algorithm = _g1_training_components(latent_dim=latent_dim, history_length=history_length)
    return StudentRunnerCfg(
        num_steps_per_env=rollout_steps,
        max_iterations=DEFAULT_STUDENT_ITERATIONS,
        save_interval=DEFAULT_SAVE_INTERVAL,
        experiment_name="g1_rickshaw_student",
        run_name="s2",
        logger="tensorboard",
        obs_groups={
            "actor": ("policy_sequence",),
            "critic": ("critic_policy", "critic"),
        },
        actor=actor,
        critic=critic,
        algorithm=algorithm,
    )


def g1_rickshaw_distillation_runner_cfg(
    *,
    latent_dim: int = DEFAULT_CONTEXT_DIM,
    history_length: int = HISTORY_LENGTH,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
) -> DistillationRunnerCfg:
    base = unitree_g1_ppo_runner_cfg()
    model = _actor_cfg(base.actor, latent_dim=latent_dim, history_length=history_length)
    return DistillationRunnerCfg(
        num_steps_per_env=rollout_steps,
        max_iterations=DEFAULT_DISTILLATION_ITERATIONS,
        save_interval=DEFAULT_SAVE_INTERVAL,
        experiment_name="g1_rickshaw_context",
        run_name="s1",
        logger="tensorboard",
        obs_groups={
            "student": ("policy_sequence",),
            "teacher": ("policy_sequence", "teacher_dynamic_sequence", "teacher_static"),
        },
        student=model,
        teacher=_actor_cfg(base.actor, latent_dim=latent_dim, history_length=history_length),
    )
