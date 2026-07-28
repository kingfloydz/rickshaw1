#!/usr/bin/env python3
"""Train or resume the privileged teacher through Mjlab's launcher."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import tyro

from _stage_training import prepare_training


@dataclass
class TeacherArgs:
    task: str = "Mjlab-G1-Rickshaw-Slopes-Teacher"
    experiment_dir: Path | None = None
    resume_checkpoint: Path | None = None
    latent_dim: int = 16
    history_length: int = 61
    rollout_steps: int = 24
    num_envs: int = 8192
    max_iterations: int = 6_000
    seed: int = 42
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])


def main() -> None:
    from mjlab.scripts.train import launch_training

    args = tyro.cli(TeacherArgs)
    cfg = prepare_training(
        task=args.task,
        experiment_dir=args.experiment_dir,
        latent_dim=args.latent_dim,
        history_length=args.history_length,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        seed=args.seed,
        gpu_ids=args.gpu_ids,
    )
    if args.resume_checkpoint is not None:
        checkpoint = args.resume_checkpoint.resolve(strict=True)
        cfg.agent.checkpoint_file = str(checkpoint)
        if args.experiment_dir is None:
            cfg = replace(cfg, log_root=str(checkpoint.parent.parent.parent))
            cfg.agent.experiment_name = checkpoint.parent.parent.name
    launch_training(args.task, cfg)


if __name__ == "__main__":
    main()
