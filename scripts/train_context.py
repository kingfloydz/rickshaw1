#!/usr/bin/env python3
"""Distill the teacher online through RSL-RL's DistillationRunner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from _stage_training import prepare_training


@dataclass
class DistillationArgs:
    teacher: Path
    task: str = "Mjlab-G1-Rickshaw-Slopes-Distillation"
    experiment_dir: Path = Path("logs/rsl_rl/g1_rickshaw_context")
    latent_dim: int = 16
    history_length: int = 61
    rollout_steps: int = 24
    num_envs: int = 8192
    max_iterations: int = 2_000
    seed: int = 42
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])


def main() -> None:
    from mjlab.scripts.train import launch_training

    args = tyro.cli(DistillationArgs)
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
    cfg.agent.teacher_checkpoint = str(args.teacher.resolve(strict=True))
    launch_training(args.task, cfg)


if __name__ == "__main__":
    main()
