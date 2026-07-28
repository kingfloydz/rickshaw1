#!/usr/bin/env python3
"""Fine-tune the distilled student with Mjlab's standard PPO runner."""

from __future__ import annotations

import argparse
from functools import partial
import os
from pathlib import Path

from _mjlab_wrappers import (
    add_project_source_to_path,
    require_existing_file,
    run_mjlab_rsl_rl,
)

add_project_source_to_path()

from g1_rickshaw_lab.policy_schema import (  # noqa: E402
    DEFAULT_CONTEXT_DIM,
    HISTORY_LENGTH,
    SUPPORTED_CONTEXT_DIMS,
    SUPPORTED_HISTORY_LENGTHS,
)
from g1_rickshaw_lab.checkpoint_handoff import initialize_student_runner  # noqa: E402
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents.rsl_rl_cfg import (  # noqa: E402
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_ROLLOUT_STEPS,
    DEFAULT_SAVE_INTERVAL,
)

DEFAULT_TASK = "Mjlab-G1-Rickshaw-Slopes-Student"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--experiment-dir", default="logs/rsl_rl/g1_rickshaw_student")
    parser.add_argument(
        "--latent-dim",
        type=int,
        choices=SUPPORTED_CONTEXT_DIMS,
        default=DEFAULT_CONTEXT_DIM,
    )
    parser.add_argument(
        "--history-length",
        type=int,
        choices=SUPPORTED_HISTORY_LENGTHS,
        default=HISTORY_LENGTH,
    )
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument(
        "--num-envs", "--num_envs", dest="num_envs", type=int, default=8192
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--seed", type=int, default=42)
    args, remaining = parser.parse_known_args()

    initialize_runner = None
    resume_arguments: list[str] = []
    experiment_name = os.fspath(Path(args.experiment_dir).resolve())
    if args.resume_checkpoint is not None:
        resume = require_existing_file(
            args.resume_checkpoint, "student resume checkpoint"
        ).resolve()
        if args.experiment_dir == "logs/rsl_rl/g1_rickshaw_student":
            experiment_name = os.fspath(resume.parent.parent)
        resume_arguments = ["--resume", "--checkpoint", os.fspath(resume)]
    else:
        if args.teacher is None or args.context is None:
            parser.error("fresh S2 training requires --teacher and --context")
        teacher = require_existing_file(args.teacher, "teacher checkpoint").resolve()
        context = require_existing_file(
            args.context, "distillation checkpoint"
        ).resolve()
        initialize_runner = partial(
            initialize_student_runner,
            teacher=teacher,
            context=context,
        )

    run_mjlab_rsl_rl(
        "train",
        [
            "--task",
            args.task,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.max_iterations),
            "--seed",
            str(args.seed),
            "--logger",
            "tensorboard",
            "--experiment_name",
            experiment_name,
            *resume_arguments,
            *remaining,
            f"agent.num_steps_per_env={args.rollout_steps}",
            f"agent.save_interval={DEFAULT_SAVE_INTERVAL}",
            f"agent.actor.latent_dim={args.latent_dim}",
            f"agent.actor.history_length={args.history_length}",
            f"env.history_length={args.history_length}",
        ],
        initialize_runner=initialize_runner,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
