"""Project-owned Mjlab/RSL-RL train and play launchers."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class PlayOptions:
    video_dir: Path | None = None
    export_policy: bool = True
    export_only: bool = False
    follow_robot_camera: bool = False
    video_name_prefix: str = "rl-video"

    def __post_init__(self) -> None:
        if self.export_only and not self.export_policy:
            raise ValueError("export_only requires export_policy")


def _parser(mode: Literal["train", "play"]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"G1 rickshaw Mjlab RSL-RL {mode} launcher")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_run", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--mimic", action="store_true")
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default=None)
    parser.add_argument("--log_project_name", default=None)
    if mode == "train":
        parser.add_argument("--video_interval", type=int, default=2000)
        parser.add_argument("--max_iterations", type=int, default=None)
        parser.add_argument("--velocity-curriculum", action="store_true")
    else:
        parser.add_argument("--real-time", action="store_true", default=False)
    return parser


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if value.lower() in {"none", "null"}:
        return None
    return value


def _set_path(root: Any, path: str, value: str) -> None:
    parts = path.split(".")
    target = root
    for part in parts[:-1]:
        target = target[part] if isinstance(target, dict) else getattr(target, part)
    leaf = parts[-1]
    current = target[leaf] if isinstance(target, dict) else getattr(target, leaf)
    converted = _coerce(value, current)
    if isinstance(target, dict):
        target[leaf] = converted
    else:
        setattr(target, leaf, converted)


def configure_history_length(env_cfg: Any, history_length: int) -> None:
    runtime = replace(env_cfg.events["initialize_task"].params["cfg"], history_length=history_length)
    for event_name in ("initialize_task", "initialize_domain", "policy_state"):
        env_cfg.events[event_name].params["cfg"] = runtime
    env_cfg.policy_update = runtime
    env_cfg.history_length = history_length
    env_cfg.observations["history"].terms["history"].params["history_length"] = history_length
    env_cfg.observations["teacher_dynamic_history"].terms["history"].params["history_length"] = history_length


def _apply_overrides(env_cfg: Any, agent_cfg: Any, overrides: list[str]) -> None:
    for token in overrides:
        if "=" not in token:
            raise ValueError(f"unsupported Mjlab override: {token}")
        key, value = token.lstrip("+").split("=", 1)
        if key == "env.history_length":
            configure_history_length(env_cfg, int(value))
        elif key.startswith("env."):
            _set_path(env_cfg, key.removeprefix("env."), value)
        elif key.startswith("agent."):
            _set_path(agent_cfg, key.removeprefix("agent."), value)
        else:
            raise ValueError(f"override must start with env. or agent.: {token}")


def run_rsl_rl(
    mode: Literal["train", "play"],
    argv: list[str],
    *,
    play_options: PlayOptions | None = None,
    initialize_runner: Callable[[Any], None] | None = None,
) -> None:
    parser = _parser(mode)
    args, overrides = parser.parse_known_args(argv)
    if mode == "train":
        _run_train(args, overrides, initialize_runner)
    else:
        if initialize_runner is not None:
            raise ValueError("runner initialization applies only to training")
        _run_play(args, overrides, play_options or PlayOptions())


def _load_configs(args: argparse.Namespace, overrides: list[str], *, play: bool):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    if not play:
        os.environ["G1_RICKSHAW_VELOCITY_CURRICULUM"] = "1" if args.velocity_curriculum else "0"
    import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.registration  # noqa: F401

    env_cfg = load_env_cfg(args.task, play=play)
    agent_cfg = load_rl_cfg(args.task)
    if args.mimic:
        from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.env_cfg import (
            enable_mimic,
        )

        enable_mimic(env_cfg)
    _apply_overrides(env_cfg, agent_cfg, overrides)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
        agent_cfg.seed = args.seed
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.logger is not None:
        agent_cfg.logger = args.logger
    if args.log_project_name is not None:
        agent_cfg.wandb_project = args.log_project_name
    if args.resume:
        agent_cfg.resume = True
    if args.load_run is not None:
        agent_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    return env_cfg, agent_cfg


def _run_train(
    args: argparse.Namespace,
    overrides: list[str],
    initialize_runner: Callable[[Any], None] | None,
) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import dump_yaml, get_checkpoint_path
    from mjlab.utils.wrappers import VideoRecorder

    env_cfg, agent_cfg = _load_configs(args, overrides, play=False)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = (log_root / run_name).resolve()
    resume_path = None
    if agent_cfg.resume:
        candidate = Path(agent_cfg.load_checkpoint)
        resume_path = (
            candidate.resolve()
            if candidate.is_file()
            else get_checkpoint_path(log_root.resolve(), agent_cfg.load_run, agent_cfg.load_checkpoint)
        )
    env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "train",
            step_trigger=lambda step: step % args.video_interval == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_type = load_runner_cls(args.task) or MjlabOnPolicyRunner
    env_cfg_dict = asdict(env_cfg)
    agent_cfg_dict = asdict(agent_cfg)
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)
    runner = runner_type(wrapped, agent_cfg_dict, str(log_dir), device)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        runner.load(str(resume_path))
        if initialize_runner is not None:
            raise ValueError("cannot combine resume loading with fresh runner initialization")
    elif initialize_runner is not None:
        initialize_runner(runner)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    wrapped.close()


def _run_play(
    args: argparse.Namespace,
    overrides: list[str],
    options: PlayOptions,
) -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import get_checkpoint_path
    from mjlab.utils.wrappers import VideoRecorder

    env_cfg, agent_cfg = _load_configs(args, overrides, play=True)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log_root = (Path("logs/rsl_rl") / agent_cfg.experiment_name).resolve()
    if args.checkpoint and Path(args.checkpoint).is_file():
        resume_path = Path(args.checkpoint).resolve()
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = resume_path.parent
    env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array" if args.video else None)
    video_dir = options.video_dir or log_dir / "videos" / "play"
    video_recorder = None
    if args.video:
        video_recorder = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            name_prefix=options.video_name_prefix,
            disable_logger=True,
        )
        env = video_recorder
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_type = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_type(wrapped, asdict(agent_cfg), None, device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)
    if options.export_policy:
        export_dir = log_dir / "exported"
        runner.export_policy_to_jit(str(export_dir), "policy.pt")
        runner.export_policy_to_onnx(str(export_dir), "policy.onnx")
    if options.export_only:
        wrapped.close()
        return
    if args.video:
        obs = wrapped.get_observations()
        for _step in range(args.video_length):
            start = time.time()
            with torch.inference_mode():
                action = policy(obs)
                obs, _, dones, _ = wrapped.step(action)
                policy.reset(dones)
            if args.real_time:
                time.sleep(max(0.0, wrapped.unwrapped.step_dt - (time.time() - start)))
    else:
        from mjlab.viewer import ViserPlayViewer

        ViserPlayViewer(wrapped.unwrapped, policy).run()
    wrapped.close()


__all__ = ["PlayOptions", "configure_history_length", "run_rsl_rl"]
