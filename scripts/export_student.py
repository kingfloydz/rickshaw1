#!/usr/bin/env python3
"""Export a student checkpoint through Mjlab's policy exporter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from _project import add_project_source

add_project_source()


@dataclass
class ExportArgs:
    checkpoint: Path
    task: str = "Mjlab-G1-Rickshaw-Slopes-Student"
    output_dir: Path | None = None
    latent_dim: int = 16
    history_length: int = 61
    device: str | None = None


def main() -> None:
    import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.registration  # noqa: F401
    from g1_rickshaw_lab.policy_schema import (
        validate_context_dim,
        validate_history_length,
    )
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.env_cfg import (
        configure_history_length,
    )
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    args = tyro.cli(ExportArgs)
    checkpoint = args.checkpoint.resolve(strict=True)
    latent_dim = validate_context_dim(args.latent_dim)
    history_length = validate_history_length(args.history_length)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    configure_history_length(env_cfg, history_length)
    env_cfg.scene.num_envs = 1
    agent_cfg.actor.latent_dim = latent_dim
    agent_cfg.actor.history_length = history_length

    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(env_cfg, device=device), clip_actions=agent_cfg.clip_actions
    )
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    output_dir = (args.output_dir or checkpoint.parent / "exported").resolve()
    runner.export_policy_to_jit(str(output_dir), "policy.pt")
    runner.export_policy_to_onnx(str(output_dir), "policy.onnx")
    env.close()


if __name__ == "__main__":
    main()
