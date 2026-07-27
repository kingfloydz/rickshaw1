"""Play or evaluate the registered task through mjlab's native CLI."""

from __future__ import annotations

import argparse
import os
import sys


PLAY_SLOPE_ENV = "G1_RICKSHAW_PLAY_SLOPE"


def _parse_rickshaw_options(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--slope",
        "--terrain-slope",
        dest="slope",
        type=float,
        default=None,
        help="Use one terrain slope, in radians, for every play environment.",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    options, remaining = _parse_rickshaw_options(sys.argv[1:])
    if options.slope is not None:
        os.environ[PLAY_SLOPE_ENV] = str(options.slope)
    sys.argv = [sys.argv[0], *remaining]

    # Task configs are materialized during registration, after the play options
    # above have been made available to their factories.
    import g1_rickshaw_lab.tasks  # noqa: F401
    from mjlab.scripts.play import main as mjlab_main

    mjlab_main()


if __name__ == "__main__":
    main()
