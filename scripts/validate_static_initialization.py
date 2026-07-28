#!/usr/bin/env python3
"""Solve, certify, and save the flat-ground MuJoCo rest pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import add_project_source

add_project_source()

from g1_rickshaw_lab.static_equilibrium import (  # noqa: E402
    STATIC_REST_POSE_PATH,
    save_mujoco_static_equilibrium,
)
from _static_equilibrium_solver import solve_mujoco_static_equilibrium  # noqa: E402
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.closed_chain import (  # noqa: E402
    build_assembled_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=STATIC_REST_POSE_PATH)
    args = parser.parse_args()
    model = build_assembled_spec().compile()
    solution = solve_mujoco_static_equilibrium(model)
    output = save_mujoco_static_equilibrium(model, solution, args.output)
    report = {
        "status": "passed",
        "rest_pose_path": str(output),
        "equality_position_error_m": solution.equality_position_error,
        "support_height_error_m": solution.support_height_error,
        "hitch_height_m": solution.hitch_height,
        "acceleration_error": solution.acceleration_error,
        "actuator_torque_ratio_max": solution.actuator_torque_ratio,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
