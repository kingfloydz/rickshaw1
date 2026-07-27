"""Mjlab lifecycle for the inclined rigid robot-rickshaw task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_from_matrix,
)

from g1_rickshaw_lab.assets.g1_dex1 import GRASP_SITE_NAMES
from g1_rickshaw_lab.assets.rickshaw import (
    BASE_LINK_NAME,
    HITCH_SITE_NAMES,
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
    WHEEL_JOINT_NAMES,
    WHEEL_LINK_NAMES,
    WHEEL_RADIUS,
)
from g1_rickshaw_lab.configuration import G1_JOINT_ORDER
from g1_rickshaw_lab.policy_schema import ACTOR_OBSERVATION_DIM, HISTORY_LENGTH, TEACHER_DYNAMIC_DIM
from g1_rickshaw_lab.static_equilibrium import MujocoStaticEquilibrium, load_mujoco_static_equilibrium

from .closed_chain import CONNECTION_NAMES, build_assembled_spec
from .mdp.dynamics import (
    RickshawKinematicState,
    combine_mass_properties,
    connect_constraint_forces,
    relative_position_in_yaw_frame,
    relative_yaw_from_quaternions,
    rickshaw_pitch_from_quaternion,
    rolling_resistance_force,
    wheel_longitudinal_slip,
)
from .mdp.events import (
    DomainRandomizationCfg,
    RickshawRuntimeState,
    _update_teacher_static_domain,
    sample_domain_parameters,
)
from .mdp.observations import ObservationHistoryState
from .mjlab_commands import rickshaw_velocity
from .sloped_reset import TERRAIN_SLOPES, build_sloped_reset_templates
from .terrain import (
    assign_terrain_types,
    terrain_frame,
    terrain_plane_poses,
    write_terrain_collision_pose,
)


@dataclass(frozen=True)
class MjlabTaskRuntimeCfg:
    domain: DomainRandomizationCfg
    history_length: int = HISTORY_LENGTH
    terrain_slope: float | None = None


def _ids(entity: Any, kind: str, names: tuple[str, ...]) -> torch.Tensor:
    finder = getattr(entity, f"find_{kind}")
    indices, resolved = finder(names, preserve_order=True)
    if tuple(resolved) != names:
        raise RuntimeError(f"{kind} order mismatch: expected {names}, got {tuple(resolved)}")
    return torch.as_tensor(indices, device=entity.data.device, dtype=torch.long)


def _body_mass_kinematics(env: Any, entity_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entity = env.scene[entity_name]
    body_ids = entity.indexing.body_ids
    masses = env.sim.model.body_mass[:, body_ids]
    total = masses.sum(dim=-1)
    weights = masses / total[:, None]
    position = torch.sum(entity.data.body_com_pos_w * weights[..., None], dim=1)
    velocity = torch.sum(entity.data.body_com_lin_vel_w * weights[..., None], dim=1)
    return position, velocity, total


def _load_static() -> tuple[Any, MujocoStaticEquilibrium]:
    model = build_assembled_spec(with_ground=True).compile()
    return model, load_mujoco_static_equilibrium(model)


@requires_model_fields("geom_pos", "geom_quat")
def initialize_mjlab_task(env: Any, env_ids: torch.Tensor | None, cfg: MjlabTaskRuntimeCfg) -> None:
    """Load the certified rest pose and allocate policy-rate state."""

    del env_ids
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.terrain_types = assign_terrain_types(
        env.num_envs,
        device=env.device,
        terrain_slope=cfg.terrain_slope,
    )
    plane_positions, plane_quaternions = terrain_plane_poses(env.scene.env_origins, env.terrain_types)
    env.terrain_geom_id = int(env.scene["terrain"].indexing.geom_ids[0])
    env.sim.model.geom_pos[ids, env.terrain_geom_id] = plane_positions
    env.sim.model.geom_quat[ids, env.terrain_geom_id] = plane_quaternions
    write_terrain_collision_pose(
        env.sim.data.geom_xpos,
        env.sim.data.geom_xmat,
        env_origins=env.scene.env_origins,
        terrain_types=env.terrain_types,
        env_ids=ids,
        geom_id=env.terrain_geom_id,
    )
    env.path_tangent_w, env.path_lateral_w, env.path_normal_w = terrain_frame(
        env.terrain_types,
        dtype=torch.float32,
    )

    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    env.policy_joint_ids = _ids(robot, "joints", G1_JOINT_ORDER)
    env.policy_actuator_ids = _ids(robot, "actuators", G1_JOINT_ORDER)
    env.wheel_joint_ids = _ids(cart, "joints", WHEEL_JOINT_NAMES)
    env.wheel_body_ids = _ids(cart, "bodies", WHEEL_LINK_NAMES)
    env.hitch_site_ids = _ids(cart, "sites", HITCH_SITE_NAMES)
    env.grasp_site_ids = _ids(robot, "sites", GRASP_SITE_NAMES)
    env.foot_body_ids = _ids(robot, "bodies", ("left_ankle_roll_link", "right_ankle_roll_link"))
    env.torso_body_id = int(_ids(robot, "bodies", ("torso_link",))[0])
    env.cart_base_body_id = int(_ids(cart, "bodies", (BASE_LINK_NAME,))[0])

    if tuple(robot.joint_names) != G1_JOINT_ORDER:
        raise RuntimeError("fixed-gripper MuJoCo robot must expose exactly the 29 policy joints")
    env._mujoco_static_model, env._mujoco_static_equilibrium = _load_static()
    model = env._mujoco_static_model
    solution = env._mujoco_static_equilibrium
    templates = build_sloped_reset_templates(model, solution.qpos, TERRAIN_SLOPES)
    terrain_types = env.terrain_types
    env.static_robot_pose = torch.as_tensor(templates.robot_root_pose, device=env.device, dtype=torch.float32)[
        terrain_types
    ]
    env.static_cart_pose = torch.as_tensor(templates.cart_root_pose, device=env.device, dtype=torch.float32)[
        terrain_types
    ]
    env.static_joint_position = torch.as_tensor(
        templates.robot_joint_position, device=env.device, dtype=torch.float32
    )[terrain_types]
    robot.data.default_joint_pos[:, env.policy_joint_ids] = env.static_joint_position
    env.static_q_ref = env.static_joint_position.clone()
    env.connection_equality_ids = torch.as_tensor(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name) for name in CONNECTION_NAMES],
        device=env.device,
        dtype=torch.long,
    )
    env.static_relative_position_b = relative_position_in_yaw_frame(
        env.static_robot_pose[:, :3], env.static_cart_pose[:, :3], env.static_cart_pose[:, 3:7]
    )
    env.static_relative_yaw = relative_yaw_from_quaternions(
        env.static_robot_pose[:, 3:7], env.static_cart_pose[:, 3:7]
    )
    env.static_rickshaw_pitch = rickshaw_pitch_from_quaternion(
        env.static_cart_pose[:, 3:7], env.path_tangent_w, env.path_normal_w
    )

    env.overspeed_margin = float(cfg.domain.calibration["safety.overspeed_margin"])
    env.rickshaw_state = RickshawRuntimeState.zeros(env.num_envs, device=env.device)
    env.observation_history_state = ObservationHistoryState.zeros(
        env.num_envs, history_length=cfg.history_length, device=env.device
    )
    env.critic_policy_observation = torch.zeros((env.num_envs, ACTOR_OBSERVATION_DIM), device=env.device)
    env.teacher_dynamic_history_state = ObservationHistoryState.zeros(
        env.num_envs,
        history_length=cfg.history_length,
        observation_dim=TEACHER_DYNAMIC_DIM,
        device=env.device,
    )
    env.hitch_height_target = float(solution.hitch_height)
    zeros = torch.zeros(env.num_envs, device=env.device)
    zeros3 = torch.zeros((env.num_envs, 3), device=env.device)
    env.rickshaw_kinematic_state = RickshawKinematicState.initialized(zeros, zeros3)
    env.cart_previous_com_velocity_w = torch.zeros((env.num_envs, 3), device=env.device)
    env.last_rolling_force_w = torch.zeros((env.num_envs, 2, 3), device=env.device)
    env.rickshaw_speed_s = zeros.clone()
    env.rickshaw_ang_vel_z = zeros.clone()
    env._mjlab_physical_state_step = -1
    env._mjlab_observation_state_step = -1


@requires_model_fields(
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "geom_friction",
    "dof_damping",
    recompute=RecomputeLevel.set_const,
)
def initialize_mjlab_domain(env: Any, env_ids: torch.Tensor | None, cfg: MjlabTaskRuntimeCfg) -> None:
    """Sample the nine startup-fixed physics parameters and write MuJoCo fields."""

    del env_ids
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    sampled = sample_domain_parameters(cfg.domain, env.num_envs, device=env.device)
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    torso_global = robot.indexing.body_ids[env.torso_body_id]
    base_global = cart.indexing.body_ids[env.cart_base_body_id]
    env_grid = ids

    default_robot_mass = env.sim.get_default_field("body_mass")[robot.indexing.body_ids]
    env._default_robot_masses_cpu = default_robot_mass[None, :].repeat(env.num_envs, 1).cpu()
    torso_mass = default_robot_mass[env.torso_body_id] + sampled["torso.mass_delta"]
    env.sim.model.body_mass[env_grid, torso_global] = torso_mass
    env.effective_torso_mass = torso_mass

    default_mass = env.sim.get_default_field("body_mass")[base_global]
    default_com = env.sim.get_default_field("body_ipos")[base_global]
    default_principal = env.sim.get_default_field("body_inertia")[base_global]
    default_quat = env.sim.get_default_field("body_iquat")[base_global]
    rotation = matrix_from_quat(default_quat)
    default_inertia = rotation @ torch.diag(default_principal) @ rotation.mT
    payload_mass = sampled["payload.mass"]
    payload_com = torch.stack((sampled["payload.com.x"], sampled["payload.com.y"], sampled["payload.com.z"]), dim=-1)
    total_mass, total_com, total_inertia = combine_mass_properties(
        default_mass.expand(env.num_envs),
        default_com.expand(env.num_envs, -1),
        default_inertia.expand(env.num_envs, -1, -1),
        payload_mass,
        payload_com,
        torch.zeros((env.num_envs, 3, 3), device=env.device),
    )
    principal, axes = torch.linalg.eigh(total_inertia)
    reflected = torch.linalg.det(axes) < 0
    axes[reflected, :, 2] *= -1
    env.sim.model.body_mass[env_grid, base_global] = total_mass
    env.sim.model.body_ipos[env_grid, base_global] = total_com
    env.sim.model.body_inertia[env_grid, base_global] = principal
    env.sim.model.body_iquat[env_grid, base_global] = quat_from_matrix(axes)
    friction = sampled["terrain.friction"]
    geom_ids = torch.cat(
        (
            robot.indexing.geom_ids,
            cart.indexing.geom_ids,
            env.scene.terrain.indexing.geom_ids,
        )
    ).to(dtype=torch.long)
    friction_grid, geom_grid = torch.meshgrid(ids, geom_ids, indexing="ij")
    env.sim.model.geom_friction[friction_grid, geom_grid, 0] = friction[:, None]
    wheel_dof_ids = cart.indexing.joint_v_adr[env.wheel_joint_ids].to(dtype=torch.long)
    dof_grid, wheel_grid = torch.meshgrid(ids, wheel_dof_ids, indexing="ij")
    wheel_damping = torch.stack((sampled["wheel.left_damping"], sampled["wheel.right_damping"]), dim=-1)
    env.sim.model.dof_damping[dof_grid, wheel_grid] = wheel_damping
    env.c_rr = sampled["rolling_resistance.c_rr"]
    cart_mass = RICKSHAW_TOTAL_MASS + payload_mass
    nominal_com = torch.tensor(RICKSHAW_CENTER_OF_MASS, device=env.device)
    cart_com = (RICKSHAW_TOTAL_MASS * nominal_com[None, :] + payload_mass[:, None] * payload_com) / cart_mass[:, None]
    env.effective_cart_mass_com = torch.cat((cart_mass[:, None], cart_com), dim=-1)
    _update_teacher_static_domain(env, cfg.domain, sampled)


def _transform_pose(env: Any, local_pose: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    local_pose = local_pose.clone()
    local_pose[:, :3] += env.scene.env_origins[env_ids]
    return local_pose


def reset_from_mujoco_static(env: Any, env_ids: torch.Tensor | None) -> None:
    """Keep the G1 root upright and incline only its ankle-pitch joints."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    # MJWarp does not recompute world-space poses for static geoms during forward().
    write_terrain_collision_pose(
        env.sim.data.geom_xpos,
        env.sim.data.geom_xmat,
        env_origins=env.scene.env_origins,
        terrain_types=env.terrain_types,
        env_ids=env_ids,
        geom_id=env.terrain_geom_id,
    )
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    model = env._mujoco_static_model

    def qadr(name: str) -> int:
        return int(model.joint(name).qposadr[0])

    qpos = torch.as_tensor(
        env._mujoco_static_equilibrium.qpos,
        device=env.device,
        dtype=torch.float32,
    ).expand(env_ids.numel(), -1)
    robot_pose = _transform_pose(env, env.static_robot_pose[env_ids], env_ids)
    cart_pose = _transform_pose(env, env.static_cart_pose[env_ids], env_ids)
    zeros6 = torch.zeros((env_ids.numel(), 6), device=env.device)
    robot.write_root_link_pose_to_sim(robot_pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(zeros6, env_ids=env_ids)
    cart.write_root_link_pose_to_sim(cart_pose, env_ids=env_ids)
    cart.write_root_link_velocity_to_sim(zeros6, env_ids=env_ids)

    robot_joint_pos = env.static_joint_position[env_ids]
    robot.write_joint_state_to_sim(robot_joint_pos, torch.zeros_like(robot_joint_pos), env_ids=env_ids)
    wheel_pos = torch.stack(
        (qpos[:, qadr("rickshaw/left_wheel_joint")], qpos[:, qadr("rickshaw/right_wheel_joint")]), dim=-1
    )
    cart.write_joint_state_to_sim(
        wheel_pos, torch.zeros_like(wheel_pos), joint_ids=env.wheel_joint_ids, env_ids=env_ids
    )
    q_ref = env.static_q_ref[env_ids]
    env.action_manager.get_term("joint_pos").set_reference(q_ref, env_ids)
    # Entity.set_joint_position_target uses direct tensor indexing, unlike the
    # write_* state helpers.  Explicitly form the env-by-joint outer product.
    robot.set_joint_position_target(
        q_ref - robot.data.encoder_bias[env_ids][:, env.policy_joint_ids],
        joint_ids=env.policy_joint_ids.unsqueeze(0),
        env_ids=env_ids.unsqueeze(1),
    )
    env.rickshaw_state.wheel_normal_force[env_ids] = 0.0
    env.rickshaw_state.wheel_longitudinal_slip[env_ids] = 0.0
    env.rickshaw_state.two_wheel_contact[env_ids] = False
    env.rickshaw_state.hand_force_w[env_ids] = 0.0
    env.rickshaw_state.connection_force_w[env_ids] = 0.0
    env.rickshaw_state.relative_position_b[env_ids] = env.static_relative_position_b[env_ids]
    env.rickshaw_state.relative_yaw[env_ids] = env.static_relative_yaw[env_ids]
    env.rickshaw_state.pitch[env_ids] = env.static_rickshaw_pitch[env_ids]
    env.observation_history_state.reset(env_ids)
    env.critic_policy_observation[env_ids] = 0.0
    env.teacher_dynamic_history_state.reset(env_ids)
    zero = torch.zeros(env_ids.numel(), device=env.device)
    env.rickshaw_kinematic_state.reset(
        zero,
        torch.zeros((env_ids.numel(), 3), device=env.device),
        env_ids,
    )
    env.cart_previous_com_velocity_w[env_ids] = 0.0
    env.last_rolling_force_w[env_ids] = 0.0
    env._mjlab_physical_state_step = -1
    env._mjlab_observation_state_step = -1


def ensure_mjlab_physical_state(env: Any) -> None:
    """Refresh all task state exactly once per policy step."""

    step = int(env.common_step_counter)
    if env._mjlab_physical_state_step == step:
        return
    env._mjlab_physical_state_step = step
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    origin = env.scene.env_origins

    hitch_position = torch.mean(cart.data.site_pos_w[:, env.hitch_site_ids], dim=1)
    env.rickshaw_state.hitch_height[:] = torch.sum((hitch_position - origin) * env.path_normal_w, dim=-1)
    pitch = rickshaw_pitch_from_quaternion(cart.data.root_link_quat_w, env.path_tangent_w, env.path_normal_w)
    env.rickshaw_state.pitch[:] = pitch
    env.rickshaw_state.relative_position_b[:] = relative_position_in_yaw_frame(
        robot.data.root_link_pos_w,
        cart.data.root_link_pos_w,
        cart.data.root_link_quat_w,
    )
    env.rickshaw_state.relative_yaw[:] = relative_yaw_from_quaternions(
        robot.data.root_link_quat_w, cart.data.root_link_quat_w
    )
    grasp_positions = robot.data.site_pos_w[:, env.grasp_site_ids]
    hitch_positions = cart.data.site_pos_w[:, env.hitch_site_ids]
    connection_position_error = torch.linalg.vector_norm(grasp_positions - hitch_positions, dim=-1)
    env.rickshaw_state.connection_residual[:] = torch.amax(connection_position_error, dim=-1)
    efc = env.sim.data.efc
    env.rickshaw_state.connection_force_w[:] = connect_constraint_forces(
        efc.type[:],
        efc.id[:],
        efc.force[:],
        env.connection_equality_ids,
        equality_constraint_type=int(mujoco.mjtConstraint.mjCNSTR_EQUALITY),
    )

    wheel_sensor = env.scene["wheel_contacts"]
    wheel_force = wheel_sensor.data.force
    wheel_normal = torch.clamp(torch.sum(wheel_force * env.path_normal_w[:, None, :], dim=-1), min=0.0)
    env.rickshaw_state.wheel_normal_force[:] = wheel_normal
    env.rickshaw_state.two_wheel_contact[:] = torch.all(wheel_normal > 1.0, dim=-1)
    _, cart_velocity, cart_mass = _body_mass_kinematics(env, "rickshaw")
    lin_vel_x, _, ang_vel_z = rickshaw_velocity(cart, env.path_normal_w)
    env.rickshaw_speed_s[:] = lin_vel_x
    env.rickshaw_ang_vel_z[:] = ang_vel_z
    env.rickshaw_kinematic_state.update(
        lin_vel_x,
        cart.data.root_link_ang_vel_w,
        env.path_lateral_w,
        env.path_normal_w,
        env.step_dt,
    )
    env.rickshaw_state.wheel_longitudinal_slip[:] = wheel_longitudinal_slip(
        cart.data.body_link_lin_vel_w[:, env.wheel_body_ids],
        cart.data.body_link_ang_vel_w[:, env.wheel_body_ids],
        env.path_tangent_w,
        env.path_normal_w,
        WHEEL_RADIUS,
    )
    acceleration = (cart_velocity - env.cart_previous_com_velocity_w) / env.step_dt
    gravity = torch.tensor((0.0, 0.0, -9.81), device=env.device)
    force_on_cart = (
        cart_mass[:, None] * acceleration
        - cart_mass[:, None] * gravity
        - torch.sum(wheel_force, dim=1)
        - torch.sum(env.last_rolling_force_w, dim=1)
    )
    valid_force = (step > 0) & torch.all(torch.isfinite(force_on_cart), dim=-1)
    force_on_cart = torch.where(
        valid_force[:, None], force_on_cart, torch.zeros_like(force_on_cart)
    )
    env.rickshaw_state.hand_force_w[:] = -force_on_cart
    env.cart_previous_com_velocity_w[:] = cart_velocity

    rolling_force = rolling_resistance_force(
        cart.data.body_link_lin_vel_w[:, env.wheel_body_ids],
        wheel_force,
        env.path_tangent_w,
        env.path_normal_w,
        env.c_rr,
    )
    env.last_rolling_force_w[:] = rolling_force
    cart.data.write_external_wrench(
        rolling_force,
        torch.zeros_like(rolling_force),
        body_ids=env.wheel_body_ids.tolist(),
    )


def advance_mjlab_policy_state(env: Any, env_ids: torch.Tensor | None, cfg: MjlabTaskRuntimeCfg) -> None:
    """Advance the online task state once per policy step."""

    del env_ids, cfg
    ensure_mjlab_physical_state(env)


__all__ = [
    "MjlabTaskRuntimeCfg",
    "advance_mjlab_policy_state",
    "ensure_mjlab_physical_state",
    "initialize_mjlab_domain",
    "initialize_mjlab_task",
    "reset_from_mujoco_static",
]
