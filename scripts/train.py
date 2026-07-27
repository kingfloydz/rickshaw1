"""Train the registered task through mjlab's native CLI."""

from __future__ import annotations

import argparse
import os
import sys


VELOCITY_CURRICULUM_ENV = "G1_RICKSHAW_VELOCITY_CURRICULUM"


def _parse_rickshaw_options(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--velocity-curriculum",
        action="store_true",
        help="Enable the staged linear and angular command velocity curriculum.",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    options, remaining = _parse_rickshaw_options(sys.argv[1:])
    os.environ[VELOCITY_CURRICULUM_ENV] = "1" if options.velocity_curriculum else "0"
    sys.argv = [sys.argv[0], *remaining]

    import g1_rickshaw_lab.tasks  # noqa: F401
    from mjlab.scripts.train import main as mjlab_main

    mjlab_main()


if __name__ == "__main__":
    main()
