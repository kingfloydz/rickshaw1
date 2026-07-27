#!/usr/bin/env python3
"""Render the certified flat-ground reset pose."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _mjlab_wrappers import add_project_source_to_path

add_project_source_to_path()

from g1_rickshaw_lab.static_equilibrium import load_mujoco_static_equilibrium  # noqa: E402
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.closed_chain import (  # noqa: E402
    build_assembled_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reset_poses"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--view", choices=("side", "front"), default="side")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")

    import mujoco

    model = build_assembled_spec(with_ground=True).compile()
    solution = load_mujoco_static_equilibrium(model)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.geomgroup[3] = 1
    if args.view == "front":
        camera.azimuth = 180.0
        camera.elevation = -5.0
        camera.distance = 3.4
    else:
        camera.azimuth = 90.0
        camera.elevation = -7.0
        camera.distance = 4.2

    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        data = mujoco.MjData(model)
        data.qpos[:] = solution.qpos
        mujoco.mj_forward(model, data)
        pelvis = np.asarray(data.body("robot/pelvis").xpos)
        cart = np.asarray(data.body("rickshaw/base_link").xpos)
        camera.lookat[:] = 0.5 * (pelvis + cart)
        camera.lookat[2] = max(0.65, camera.lookat[2])
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        label = (
            f"hitch height {solution.hitch_height:.3f} m   "
            f"torque ratio {solution.actuator_torque_ratio:.3f}"
        )
        draw.rectangle((16, 14, 530, 48), fill=(255, 255, 255, 220))
        draw.text((25, 23), label, fill=(20, 20, 20))
        output_name = "reset_pose_front.png" if args.view == "front" else "reset_pose_flat.png"
        output = output_dir / output_name
        image.save(output)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
