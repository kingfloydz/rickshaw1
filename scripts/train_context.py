#!/usr/bin/env python3
"""Train the S1 student with RSL-RL 5.4.0 Distillation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import random
import time
from typing import Any

from _mjlab_wrappers import (
    add_project_source_to_path,
    load_mjlab_configs,
    require_existing_file,
)

add_project_source_to_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.algorithms import Distillation  # noqa: E402
from rsl_rl.storage import RolloutStorage  # noqa: E402

from g1_rickshaw_lab.provenance import (  # noqa: E402
    extract_checkpoint_metadata,
    save_checkpoint_atomic,
)
from g1_rickshaw_lab.policy_schema import ACTION_DIM  # noqa: E402
from g1_rickshaw_lab.rl.rsl_rl_models import RslRickshawActorModel  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    CHECKPOINT_LINEAGE_KEY,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_STAGE_KEY,
    GUIDE_MAX_ITERATIONS,
    GUIDE_TRAINING_NUM_ENVS,
    GUIDE_TRAINING_PARAMETERS,
    S1_DETERMINISTIC_ALGORITHMS,
    TRAINING_ARTIFACT_INTERVAL,
    TRAINING_CONFIGURATION_KEY,
    build_training_configuration,
    load_stage_checkpoint,
    require_pinned_rsl_rl,
    training_mimic_enabled,
    validate_guide_training_configuration,
)


DEFAULT_TASK = "Mjlab-G1-Rickshaw-Slopes-Teacher"
DEFAULT_OUTPUT = "logs/rsl_rl/g1_rickshaw_context/s1_context.pt"
S1_GUIDE_PARAMETERS = GUIDE_TRAINING_PARAMETERS["s1_context_distillation"]


def seed_s1_training(seed: int) -> None:
    """Seed the environment, student initialization, and action sampling."""

    if isinstance(seed, bool) or seed < 0 or seed > 2**32 - 1:
        raise ValueError("S1 training seed must lie in [0, 2**32-1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(S1_DETERMINISTIC_ALGORITHMS)
    torch.backends.cudnn.deterministic = S1_DETERMINISTIC_ALGORITHMS
    torch.backends.cudnn.benchmark = not S1_DETERMINISTIC_ALGORITHMS


def initialize_student_from_teacher(
    student: RslRickshawActorModel,
    teacher: RslRickshawActorModel,
) -> None:
    """Initialize the shared actor and observation normalizer from S0."""

    student.policy.load_state_dict(teacher.policy.state_dict(), strict=True)
    student.policy_obs_normalizer.load_state_dict(
        teacher.policy_obs_normalizer.state_dict(), strict=True
    )


def _saved_environment_step(checkpoint: Mapping[str, Any]) -> int:
    infos = checkpoint.get("infos")
    env_state = infos.get("env_state") if isinstance(infos, Mapping) else None
    value = env_state.get("common_step_counter") if isinstance(env_state, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("S0 teacher checkpoint is missing its environment step")
    return value


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_envs",
        "max_iterations",
        "num_learning_epochs",
        "gradient_length",
        "log_interval",
        "save_interval",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if args.max_grad_norm is not None and (
        not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0
    ):
        raise ValueError("max_grad_norm must be positive and finite")
    if isinstance(args.training_seed, bool) or not 0 <= args.training_seed <= 2**32 - 1:
        raise ValueError("training_seed must lie in [0, 2**32-1]")


def _checkpoint(
    *,
    algorithm: Distillation,
    training_configuration: Mapping[str, Any],
    teacher_path: Path,
    teacher_curriculum_iteration: int,
    common_step_counter: int,
    completed_iterations: int,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    checkpoint = algorithm.save()
    checkpoint.update({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_STAGE_KEY: "s1_context_distillation",
        CHECKPOINT_CURRICULUM_ITERATION_KEY: teacher_curriculum_iteration,
        "iter": completed_iterations,
        "infos": {
            "env_state": {"common_step_counter": common_step_counter},
        },
        TRAINING_CONFIGURATION_KEY: dict(training_configuration),
        "training": {
            "completed_iterations": completed_iterations,
            "algorithm": "rsl_rl.algorithms.Distillation",
        },
        "metrics": dict(metrics),
        CHECKPOINT_LINEAGE_KEY: {
            "teacher_checkpoint": os.fspath(teacher_path),
        },
    })
    return checkpoint


def train(args: argparse.Namespace) -> Path:  # noqa: C901
    require_pinned_rsl_rl()
    seed_s1_training(args.training_seed)
    teacher_path = require_existing_file(args.teacher, "teacher checkpoint").resolve()
    teacher_checkpoint = load_stage_checkpoint(
        teacher_path,
        expected_stage="s0_teacher",
        validate_runtime=True,
    )
    teacher_configuration = dict(teacher_checkpoint[TRAINING_CONFIGURATION_KEY])
    if teacher_configuration["task"] != args.task:
        raise ValueError("S1 task differs from the S0 teacher training task")
    parameters = teacher_configuration["training_parameters"]
    latent_dim = int(parameters["latent_dim"])
    history_length = int(parameters["history_length"])
    rollout_steps = int(parameters["rollout_steps"])
    mimic = training_mimic_enabled(teacher_configuration)
    teacher_curriculum_iteration = teacher_checkpoint[
        CHECKPOINT_CURRICULUM_ITERATION_KEY
    ]
    common_step_counter = _saved_environment_step(teacher_checkpoint)
    metadata = extract_checkpoint_metadata(teacher_checkpoint)
    stage_coverage = {
        "TRAINING": args.num_envs * rollout_steps * args.max_iterations,
    }
    training_configuration = build_training_configuration(
        stage="s1_context_distillation",
        task=args.task,
        num_envs=args.num_envs,
        seed=args.training_seed,
        max_iterations=args.max_iterations,
        guide_parameters=S1_GUIDE_PARAMETERS,
        resolved_parameters={
            "algorithm": "rsl_rl.algorithms.Distillation",
            "optimizer": "adam",
            "loss_type": args.loss_type,
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
            "num_learning_epochs": args.num_learning_epochs,
            "gradient_length": args.gradient_length,
            "num_steps_per_env": rollout_steps,
            "student_actions_drive_environment": True,
            "teacher_target": "deterministic_action_mean",
            "mimic": mimic,
        },
        actor_initialized_from_teacher=True,
        stage_coverage=stage_coverage,
        latent_dim=latent_dim,
        history_length=history_length,
        rollout_steps=rollout_steps,
    )
    validate_guide_training_configuration(
        training_configuration,
        expected_stage="s1_context_distillation",
    )

    if args.device == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    output = Path(args.output).resolve()
    env = None
    try:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from rsl_rl.utils import check_nan

        env_cfg, _ = load_mjlab_configs(
            args.task,
            play=False,
            num_envs=args.num_envs,
            seed=args.training_seed,
            history_length=history_length,
            mimic=mimic,
        )
        raw_env = ManagerBasedRlEnv(env_cfg, device=device_name)
        raw_env.common_step_counter = common_step_counter
        env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
        observation, _ = env.reset()
        observation = observation.to(device)
        teacher = RslRickshawActorModel(
            observation,
            {
                "teacher": [
                    "policy",
                    "history",
                    "teacher_dynamic_history",
                    "teacher_static",
                ]
            },
            "teacher",
            ACTION_DIM,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution"},
            latent_dim=latent_dim,
            history_length=history_length,
        ).to(device)
        student = RslRickshawActorModel(
            observation,
            {"student": ["policy", "history"]},
            "student",
            ACTION_DIM,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution"},
            latent_dim=latent_dim,
            history_length=history_length,
        ).to(device)
        storage = RolloutStorage(
            "distillation",
            args.num_envs,
            rollout_steps,
            observation,
            [ACTION_DIM],
            device_name,
        )
        algorithm = Distillation(
            student,
            teacher,
            storage,
            num_learning_epochs=args.num_learning_epochs,
            gradient_length=args.gradient_length,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            loss_type=args.loss_type,
            optimizer="adam",
            device=device_name,
        )
        algorithm.load(
            teacher_checkpoint,
            load_cfg={
                "student": False,
                "teacher": True,
                "optimizer": False,
                "iteration": False,
            },
            strict=True,
        )
        initialize_student_from_teacher(student, teacher)
        algorithm.train_mode()
        env.episode_length_buf = torch.randint_like(
            env.episode_length_buf,
            high=int(env.max_episode_length),
        )
        observation = observation.to(device)
        last_metrics: dict[str, float] = {}
        start_time = time.perf_counter()
        for iteration in range(1, args.max_iterations + 1):
            reward_sum = torch.zeros((), device=device)
            with torch.inference_mode():
                for _ in range(rollout_steps):
                    actions = algorithm.act(observation)
                    observation, rewards, dones, extras = env.step(
                        actions.to(env.device)
                    )
                    check_nan(observation, rewards, dones)
                    observation, rewards, dones = (
                        observation.to(device),
                        rewards.to(device),
                        dones.to(device),
                    )
                    algorithm.process_env_step(
                        observation, rewards, dones, extras
                    )
                    reward_sum += rewards.mean()
            algorithm.compute_returns(observation)
            last_metrics = algorithm.update()
            last_metrics["mean_reward"] = float(reward_sum / rollout_steps)
            if iteration % args.log_interval == 0:
                elapsed = time.perf_counter() - start_time
                steps = iteration * rollout_steps * args.num_envs
                print(
                    f"iter={iteration} behavior={last_metrics['behavior']:.6f} "
                    f"reward={last_metrics['mean_reward']:.3f} "
                    f"steps_per_second={steps / elapsed:.0f}"
                )
            if iteration % args.save_interval == 0 or iteration == args.max_iterations:
                payload = _checkpoint(
                    algorithm=algorithm,
                    training_configuration=training_configuration,
                    teacher_path=teacher_path,
                    teacher_curriculum_iteration=teacher_curriculum_iteration,
                    common_step_counter=int(raw_env.common_step_counter),
                    completed_iterations=iteration,
                    metrics=last_metrics,
                )
                save_checkpoint_atomic(payload, output, metadata=metadata)
    finally:
        if env is not None:
            env.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=GUIDE_TRAINING_NUM_ENVS,
    )
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=GUIDE_MAX_ITERATIONS["s1_context_distillation"],
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--num-learning-epochs", type=int, default=1)
    parser.add_argument("--gradient-length", type=int, default=15)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="mse")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=TRAINING_ARTIFACT_INTERVAL,
    )
    args = parser.parse_args()
    _validate_args(args)
    output = train(args)
    print(f"saved S1 student checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
