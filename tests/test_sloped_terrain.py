from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from g1_rickshaw_lab.assets.mujoco_spec import ALL_COLLISION_BITS, GROUND_COLLISION_BIT
from g1_rickshaw_lab.configuration import G1_JOINT_ORDER
from g1_rickshaw_lab.static_equilibrium import load_mujoco_static_equilibrium
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.closed_chain import (
    build_assembled_spec,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.sloped_reset import (
    build_sloped_reset_templates,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.terrain import (
    TERRAIN_SLOPES,
    assign_terrain_types,
    terrain_frame,
    terrain_plane_poses,
    terrain_type_for_slope,
    write_terrain_collision_pose,
)


def test_terrain_has_nineteen_evenly_assigned_slopes() -> None:
    assert TERRAIN_SLOPES == tuple(index / 100.0 for index in range(-8, 11))
    terrain_types = assign_terrain_types(8192, device="cpu")
    counts = torch.bincount(terrain_types, minlength=len(TERRAIN_SLOPES))
    assert terrain_types[0] == 0
    assert terrain_types[-1] == len(TERRAIN_SLOPES) - 1
    assert int(counts.max() - counts.min()) <= 1


def test_terrain_can_assign_every_environment_to_one_slope() -> None:
    terrain_types = assign_terrain_types(8, device="cpu", terrain_slope=0.05)

    assert terrain_type_for_slope(0.05) == 13
    assert terrain_types.tolist() == [13] * 8
    with pytest.raises(ValueError, match="must be one of"):
        assign_terrain_types(1, device="cpu", terrain_slope=0.055)


def test_terrain_plane_poses_and_frames_match_each_slope() -> None:
    terrain_types = torch.arange(len(TERRAIN_SLOPES))
    origins = torch.arange(3 * len(TERRAIN_SLOPES), dtype=torch.float64).reshape(-1, 3)
    positions, quaternions = terrain_plane_poses(origins, terrain_types)
    slopes = torch.as_tensor(TERRAIN_SLOPES, dtype=torch.float64)
    expected_quaternions = torch.stack(
        (
            torch.cos(-0.5 * slopes),
            torch.zeros_like(slopes),
            torch.sin(-0.5 * slopes),
            torch.zeros_like(slopes),
        ),
        dim=-1,
    )
    torch.testing.assert_close(positions, origins)
    torch.testing.assert_close(quaternions, expected_quaternions)

    tangent, lateral, normal = terrain_frame(terrain_types, dtype=torch.float64)
    torch.testing.assert_close(
        lateral, torch.eye(3, dtype=torch.float64)[1].expand_as(lateral)
    )
    torch.testing.assert_close(
        torch.sum(tangent * normal, dim=-1), torch.zeros_like(slopes)
    )
    torch.testing.assert_close(torch.cross(tangent, lateral, dim=-1), normal)


def test_terrain_collision_pose_updates_selected_mjwarp_worlds() -> None:
    env_origins = torch.tensor(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (6.0, 0.0, 0.0)),
        dtype=torch.float64,
    )
    terrain_types = torch.tensor((0, 8, 18))
    env_ids = torch.tensor((0, 2))
    geom_xpos = torch.full((3, 2, 3), torch.nan, dtype=torch.float64)
    geom_xmat = torch.full((3, 2, 3, 3), torch.nan, dtype=torch.float64)

    write_terrain_collision_pose(
        geom_xpos,
        geom_xmat,
        env_origins=env_origins,
        terrain_types=terrain_types,
        env_ids=env_ids,
        geom_id=1,
    )

    tangent, lateral, normal = terrain_frame(
        terrain_types[env_ids], dtype=torch.float64
    )
    torch.testing.assert_close(geom_xpos[env_ids, 1], env_origins[env_ids])
    torch.testing.assert_close(
        geom_xmat[env_ids, 1],
        torch.stack((tangent, lateral, normal), dim=-1),
    )
    assert torch.isnan(geom_xpos[1]).all()
    assert torch.isnan(geom_xmat[1]).all()


def test_mjwarp_static_plane_uses_written_world_pose_for_collision() -> None:
    mjwarp = pytest.importorskip("mujoco_warp")
    warp = pytest.importorskip("warp")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="terrain">
              <geom name="terrain" type="plane" size="0 0 0.1"/>
            </body>
            <body name="ball" pos="5 0 0.58">
              <freejoint/>
              <geom name="ball" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with warp.ScopedDevice("cpu"):
        warp_model = mjwarp.put_model(model)
        warp_data = mjwarp.put_data(model, data, nworld=2, nconmax=8, njmax=16)
        terrain_geom_id = model.geom("terrain").id
        geom_xpos = warp.to_torch(warp_data.geom_xpos)
        geom_xmat = warp.to_torch(warp_data.geom_xmat)
        write_terrain_collision_pose(
            geom_xpos,
            geom_xmat,
            env_origins=torch.zeros((2, 3)),
            terrain_types=torch.tensor((8, 18)),
            env_ids=torch.tensor((1,)),
            geom_id=terrain_geom_id,
        )
        expected_matrix = geom_xmat[1, terrain_geom_id].clone()

        mjwarp.forward(warp_model, warp_data)

        torch.testing.assert_close(geom_xmat[1, terrain_geom_id], expected_matrix)
        assert int(warp_data.nacon.numpy()[0]) == 1
        assert int(warp_data.contact.worldid.numpy()[0]) == 1
        _, _, expected_normal = terrain_frame(torch.tensor((18,)), dtype=torch.float32)
        np.testing.assert_allclose(
            warp_data.contact.frame.numpy()[0, 0],
            expected_normal[0].numpy(),
            atol=1.0e-6,
        )


def test_each_cylinder_wheel_has_two_plane_contacts_on_every_slope() -> None:
    flat_model = build_assembled_spec(with_ground=True).compile()
    flat_qpos = load_mujoco_static_equilibrium(flat_model).qpos
    templates = build_sloped_reset_templates(flat_model, flat_qpos, TERRAIN_SLOPES)

    spec = build_assembled_spec(with_ground=False)
    terrain_body = spec.worldbody.add_body(name="terrain_body")
    terrain_geom = terrain_body.add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05),
    )
    terrain_geom.contype = GROUND_COLLISION_BIT
    terrain_geom.conaffinity = ALL_COLLISION_BITS
    terrain_geom.friction[:3] = (1.0, 0.005, 0.0001)
    model = spec.compile()

    robot_root = int(model.joint("robot/floating_base_joint").qposadr[0])
    cart_root = int(model.joint("rickshaw/floating_base_joint").qposadr[0])
    joint_qpos = [
        int(model.joint(f"robot/{name}").qposadr[0]) for name in G1_JOINT_ORDER
    ]
    terrain_body_id = model.body("terrain_body").id
    terrain_geom_id = model.geom("terrain").id
    wheel_body_ids = [
        model.body(f"rickshaw/{side}_wheel_link").id for side in ("left", "right")
    ]

    for slope_index, slope in enumerate(TERRAIN_SLOPES):
        model.body_quat[terrain_body_id] = (
            np.cos(-0.5 * slope),
            0.0,
            np.sin(-0.5 * slope),
            0.0,
        )
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        data.qpos[:] = flat_qpos
        data.qpos[robot_root : robot_root + 7] = templates.robot_root_pose[slope_index]
        data.qpos[cart_root : cart_root + 7] = templates.cart_root_pose[slope_index]
        data.qpos[joint_qpos] = templates.robot_joint_position[slope_index]
        mujoco.mj_forward(model, data)

        wheel_contacts = [0, 0]
        for contact in data.contact[: data.ncon]:
            if terrain_geom_id not in (contact.geom1, contact.geom2):
                continue
            other_geom = (
                contact.geom2 if contact.geom1 == terrain_geom_id else contact.geom1
            )
            other_body = model.geom_bodyid[other_geom]
            for wheel_index, wheel_body in enumerate(wheel_body_ids):
                wheel_contacts[wheel_index] += int(other_body == wheel_body)
        assert wheel_contacts == [2, 2], f"slope={slope}: contacts={wheel_contacts}"
