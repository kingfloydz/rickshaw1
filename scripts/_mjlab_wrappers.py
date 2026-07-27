"""Shared Mjlab helpers for project command-line workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from g1_rickshaw_lab.rl.runner import RunnerContext
    from g1_rickshaw_lab.workflows.rsl_rl import PlayOptions

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"


def add_project_source_to_path() -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))


def require_existing_file(path: str | Path, label: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def load_mjlab_configs(
    task: str,
    *,
    play: bool,
    num_envs: int,
    seed: int,
    history_length: int,
    mimic: bool = False,
) -> tuple[Any, Any]:
    """Load a registered Mjlab environment/runner pair with shared dimensions."""

    import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.registration  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from g1_rickshaw_lab.workflows.rsl_rl import configure_history_length

    env_cfg = load_env_cfg(task, play=play)
    if mimic:
        from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.env_cfg import (
            enable_mimic,
        )

        enable_mimic(env_cfg)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    configure_history_length(env_cfg, history_length)
    agent_cfg = load_rl_cfg(task)
    agent_cfg.seed = seed
    agent_cfg.actor.history_length = history_length
    return env_cfg, agent_cfg


def run_mjlab_rsl_rl(
    mode: Literal["train", "play"],
    argv: list[str],
    *,
    runner_context: RunnerContext,
    play_options: PlayOptions | None = None,
) -> None:
    add_project_source_to_path()
    from g1_rickshaw_lab.workflows.rsl_rl import run_rsl_rl

    run_rsl_rl(mode, argv, runner_context=runner_context, play_options=play_options)


__all__ = [
    "add_project_source_to_path",
    "load_mjlab_configs",
    "require_existing_file",
    "run_mjlab_rsl_rl",
]
