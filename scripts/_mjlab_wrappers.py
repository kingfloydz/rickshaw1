"""Shared Mjlab helpers for project command-line workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
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


def run_mjlab_rsl_rl(
    mode: Literal["train", "play"],
    argv: list[str],
    *,
    play_options: PlayOptions | None = None,
    initialize_runner: Callable[[Any], None] | None = None,
) -> None:
    add_project_source_to_path()
    from g1_rickshaw_lab.workflows.rsl_rl import run_rsl_rl

    run_rsl_rl(
        mode,
        argv,
        play_options=play_options,
        initialize_runner=initialize_runner,
    )


__all__ = [
    "add_project_source_to_path",
    "require_existing_file",
    "run_mjlab_rsl_rl",
]
