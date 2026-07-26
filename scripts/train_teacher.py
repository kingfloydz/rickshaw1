#!/usr/bin/env python3
"""Train the S0 privileged teacher with Mjlab RSL-RL."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from _mjlab_wrappers import (
    add_project_source_to_path,
    require_existing_file,
    run_mjlab_rsl_rl,
)

add_project_source_to_path()

from g1_rickshaw_lab.rl.runner import RunnerContext  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    DEFAULT_TRAINING_PARAMETERS,
    GUIDE_TRAINING_NUM_ENVS,
    GUIDE_TRAINING_PARAMETERS,
    GUIDE_TRAINING_TASK,
    SUPPORTED_CONTEXT_DIMS,
    SUPPORTED_HISTORY_LENGTHS,
    SUPPORTED_ROLLOUT_STEPS,
    TRAINING_CONFIGURATION_KEY,
    build_training_configuration,
    cli_value,
    guide_max_iterations,
    load_s0_resume_checkpoint,
    require_pinned_rsl_rl,
    training_artifact_interval,
    training_mimic_enabled,
)

DEFAULT_TASK = GUIDE_TRAINING_TASK
S0_GUIDE_PARAMETERS = GUIDE_TRAINING_PARAMETERS["s0_teacher"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Optional isolated RSL-RL experiment root for a fresh S0 run.",
    )
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--latent-dim", type=int, choices=SUPPORTED_CONTEXT_DIMS, default=None)
    parser.add_argument("--history-length", type=int, choices=SUPPORTED_HISTORY_LENGTHS, default=None)
    parser.add_argument("--rollout-steps", type=int, choices=SUPPORTED_ROLLOUT_STEPS, default=None)
    parser.add_argument(
        "--mimic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=GUIDE_TRAINING_NUM_ENVS,
    )
    args, remaining = parser.parse_known_args()
    owned_resume_flags = ("--resume", "--load_run", "--checkpoint", "--agent")
    if args.resume_checkpoint is not None:
        owned_resume_flags += ("--experiment_name",)
    if any(token == flag or token.startswith(flag + "=") for token in remaining for flag in owned_resume_flags):
        raise ValueError("S0 owns its agent and resume selection; use --resume-checkpoint")
    require_pinned_rsl_rl()
    resume_path: Path | None = None
    resume_configuration = None
    if args.resume_checkpoint is not None:
        resume_path = require_existing_file(
            args.resume_checkpoint,
            "S0 resume checkpoint",
        ).resolve()
        loaded = load_s0_resume_checkpoint(
            resume_path,
            validate_runtime=True,
        )
        resume_configuration = loaded[TRAINING_CONFIGURATION_KEY]
    resume_parameters = None if resume_configuration is None else resume_configuration["training_parameters"]
    resume_mimic = (
        False
        if resume_configuration is None
        else training_mimic_enabled(resume_configuration)
    )
    mimic = resume_mimic if args.mimic is None else args.mimic
    if resume_configuration is not None and mimic != resume_mimic:
        raise ValueError("S0 resume cannot change mimic")
    defaults = DEFAULT_TRAINING_PARAMETERS if resume_parameters is None else resume_parameters
    latent_dim = int(defaults["latent_dim"] if args.latent_dim is None else args.latent_dim)
    history_length = int(defaults["history_length"] if args.history_length is None else args.history_length)
    rollout_steps = int(defaults["rollout_steps"] if args.rollout_steps is None else args.rollout_steps)
    if resume_parameters is not None and (
        latent_dim != int(resume_parameters["latent_dim"])
        or history_length != int(resume_parameters["history_length"])
        or rollout_steps != int(resume_parameters["rollout_steps"])
    ):
        raise ValueError("S0 resume cannot change latent_dim, history_length, or rollout_steps")
    seed = cli_value(
        remaining,
        "--seed",
        hydra_keys=("agent.seed",),
        default=42,
        cast=int,
    )
    max_iterations = cli_value(
        remaining,
        "--max_iterations",
        hydra_keys=("agent.max_iterations",),
        default=guide_max_iterations("s0_teacher", rollout_steps),
        cast=int,
    )
    training_configuration = build_training_configuration(
        stage="s0_teacher",
        task=args.task,
        num_envs=args.num_envs,
        seed=seed,
        max_iterations=max_iterations,
        guide_parameters=S0_GUIDE_PARAMETERS,
        resolved_parameters={
            "seed": seed,
            "max_iterations": max_iterations,
            "num_envs": args.num_envs,
            "latent_dim": latent_dim,
            "num_steps_per_env": rollout_steps,
            "save_interval": training_artifact_interval(rollout_steps),
            "mimic": mimic,
            "launcher_arguments": list(remaining),
        },
        actor_initialized_from_teacher=None,
        stage_coverage=None,
        latent_dim=latent_dim,
        history_length=history_length,
        rollout_steps=rollout_steps,
    )
    runner_context = RunnerContext.training(
        stage="s0_teacher",
        training_configuration=training_configuration,
    )
    runtime_overrides = [
        f"agent.num_steps_per_env={rollout_steps}",
        f"agent.save_interval={training_artifact_interval(rollout_steps)}",
        f"agent.actor.latent_dim={latent_dim}",
        f"agent.actor.history_length={history_length}",
        f"env.history_length={history_length}",
    ]
    experiment_arguments: list[str] = []
    mimic_arguments = ["--mimic"] if mimic else []
    if resume_path is not None:
        experiment_root = resume_path.parent.parent
        if args.experiment_dir is not None and Path(args.experiment_dir).resolve() != experiment_root.resolve():
            raise ValueError("--experiment-dir must match the resume checkpoint experiment root")
        experiment_arguments = [
            "--experiment_name",
            os.fspath(experiment_root),
            "--resume",
            "--load_run",
            "^" + re.escape(resume_path.parent.name) + "$",
            "--checkpoint",
            "^" + re.escape(resume_path.name) + "$",
        ]
    elif args.experiment_dir is not None:
        experiment_arguments = [
            "--experiment_name",
            os.fspath(Path(args.experiment_dir).resolve()),
        ]
    run_mjlab_rsl_rl(
        "train",
        [
            "--task",
            args.task,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(max_iterations),
            "--seed",
            str(seed),
            "--logger",
            "tensorboard",
            *mimic_arguments,
            *experiment_arguments,
            *remaining,
            *runtime_overrides,
        ],
        runner_context=runner_context,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
