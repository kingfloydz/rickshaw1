"""Shared stage configuration before delegating to Mjlab's trainer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from _project import add_project_source

add_project_source()

from g1_rickshaw_lab.policy_schema import validate_context_dim, validate_history_length  # noqa: E402
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.env_cfg import (  # noqa: E402
    configure_history_length,
)


def prepare_training(
    *,
    task: str,
    experiment_dir: Path | None,
    latent_dim: int,
    history_length: int,
    rollout_steps: int,
    num_envs: int,
    max_iterations: int,
    seed: int,
    gpu_ids: list[int] | Literal["all"] | None,
):
    import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.registration  # noqa: F401
    from mjlab.scripts.train import TrainConfig

    latent_dim = validate_context_dim(latent_dim)
    history_length = validate_history_length(history_length)
    cfg = TrainConfig.from_task(task)
    configure_history_length(cfg.env, history_length)
    cfg.env.scene.num_envs = num_envs
    cfg.env.seed = seed
    cfg.agent.seed = seed
    cfg.agent.num_steps_per_env = rollout_steps
    cfg.agent.max_iterations = max_iterations

    for model_name in ("actor", "student", "teacher"):
        model = getattr(cfg.agent, model_name, None)
        if model is not None:
            model.latent_dim = latent_dim
            model.history_length = history_length

    log_root = cfg.log_root
    if experiment_dir is not None:
        directory = experiment_dir.resolve()
        log_root = str(directory.parent)
        cfg.agent.experiment_name = directory.name
    return replace(cfg, log_root=log_root, gpu_ids=gpu_ids)


__all__ = ["prepare_training"]
