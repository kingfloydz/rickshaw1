#!/usr/bin/env python3
"""Play or export a student checkpoint with Mjlab's standard runner."""

from __future__ import annotations

import argparse
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
from g1_rickshaw_lab.workflows.rsl_rl import PlayOptions  # noqa: E402

DEFAULT_TASK = "Mjlab-G1-Rickshaw-Slopes-Student"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--export-only", action="store_true")
    args, remaining = parser.parse_known_args()

    checkpoint = require_existing_file(args.checkpoint, "student checkpoint").resolve()
    run_mjlab_rsl_rl(
        "play",
        [
            "--task",
            args.task,
            "--checkpoint",
            str(checkpoint),
            f"agent.actor.latent_dim={args.latent_dim}",
            f"agent.actor.history_length={args.history_length}",
            f"env.history_length={args.history_length}",
            *remaining,
        ],
        play_options=PlayOptions(
            video_dir=None
            if args.video_dir is None
            else Path(args.video_dir).resolve(),
            export_only=args.export_only,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
