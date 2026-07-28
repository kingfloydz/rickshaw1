"""RSL-RL configurations for teacher, distillation, and student training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
    RslRlBaseRunnerCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

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
    rnd_cfg: dict[str, Any] | None = None
    symmetry_cfg: dict[str, Any] | None = None


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "DistillationRunner"
    student: RickshawActorCfg = field(default_factory=RickshawActorCfg)
    teacher: RickshawActorCfg = field(default_factory=RickshawActorCfg)
    algorithm: DistillationAlgorithmCfg = field(default_factory=DistillationAlgorithmCfg)
    teacher_checkpoint: str | None = None


def _actor_cfg(*, latent_dim: int, history_length: int) -> RickshawActorCfg:
    return RickshawActorCfg(
        class_name="g1_rickshaw_lab.rl.rsl_rl_models:RslRickshawActorModel",
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        latent_dim=latent_dim,
        history_length=history_length,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def _critic_cfg() -> RslRlModelCfg:
    return RslRlModelCfg(
        class_name="g1_rickshaw_lab.rl.rsl_rl_models:RslRickshawCriticModel",
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )


def _ppo_cfg() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def g1_rickshaw_teacher_ppo_runner_cfg(
    *,
    latent_dim: int = DEFAULT_CONTEXT_DIM,
    history_length: int = HISTORY_LENGTH,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
) -> TeacherRunnerCfg:
    return TeacherRunnerCfg(
        seed=42,
        num_steps_per_env=rollout_steps,
        max_iterations=DEFAULT_TEACHER_ITERATIONS,
        save_interval=DEFAULT_SAVE_INTERVAL,
        experiment_name="g1_rickshaw_teacher",
        run_name="s0",
        logger="tensorboard",
        clip_actions=None,
        obs_groups={
            "actor": ("policy", "history", "teacher_dynamic_history", "teacher_static"),
            "critic": ("critic_policy", "critic"),
        },
        actor=_actor_cfg(latent_dim=latent_dim, history_length=history_length),
        critic=_critic_cfg(),
        algorithm=_ppo_cfg(),
    )


def g1_rickshaw_student_ppo_runner_cfg(
    *,
    latent_dim: int = DEFAULT_CONTEXT_DIM,
    history_length: int = HISTORY_LENGTH,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
) -> StudentRunnerCfg:
    return StudentRunnerCfg(
        seed=42,
        num_steps_per_env=rollout_steps,
        max_iterations=DEFAULT_STUDENT_ITERATIONS,
        save_interval=DEFAULT_SAVE_INTERVAL,
        experiment_name="g1_rickshaw_student",
        run_name="s2",
        logger="tensorboard",
        clip_actions=None,
        obs_groups={
            "actor": ("policy", "history"),
            "critic": ("critic_policy", "critic"),
        },
        actor=_actor_cfg(latent_dim=latent_dim, history_length=history_length),
        critic=_critic_cfg(),
        algorithm=_ppo_cfg(),
    )


def g1_rickshaw_distillation_runner_cfg(
    *,
    latent_dim: int = DEFAULT_CONTEXT_DIM,
    history_length: int = HISTORY_LENGTH,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
) -> DistillationRunnerCfg:
    model = _actor_cfg(latent_dim=latent_dim, history_length=history_length)
    return DistillationRunnerCfg(
        seed=42,
        num_steps_per_env=rollout_steps,
        max_iterations=DEFAULT_DISTILLATION_ITERATIONS,
        save_interval=DEFAULT_SAVE_INTERVAL,
        experiment_name="g1_rickshaw_context",
        run_name="s1",
        logger="tensorboard",
        clip_actions=None,
        obs_groups={
            "student": ("policy", "history"),
            "teacher": ("policy", "history", "teacher_dynamic_history", "teacher_static"),
        },
        student=model,
        teacher=_actor_cfg(latent_dim=latent_dim, history_length=history_length),
    )


__all__ = [
    "DEFAULT_DISTILLATION_ITERATIONS",
    "DEFAULT_ROLLOUT_STEPS",
    "DEFAULT_SAVE_INTERVAL",
    "DEFAULT_STUDENT_ITERATIONS",
    "DEFAULT_TEACHER_ITERATIONS",
    "DistillationRunnerCfg",
    "RickshawActorCfg",
    "StudentRunnerCfg",
    "TeacherRunnerCfg",
    "g1_rickshaw_distillation_runner_cfg",
    "g1_rickshaw_student_ppo_runner_cfg",
    "g1_rickshaw_teacher_ppo_runner_cfg",
]
