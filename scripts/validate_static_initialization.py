#!/usr/bin/env python3
"""Solve, certify, and save the flat-ground MuJoCo rest pose."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g1_rickshaw_lab.static_equilibrium import (  # noqa: E402
    STATIC_REST_POSE_PATH,
    save_mujoco_static_equilibrium,
    solve_mujoco_static_equilibrium,
)
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
        "fat2_reference_angle_rad": solution.fat2_reference_angle,
        "actuator_torque_ratio_max": solution.actuator_torque_ratio,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
