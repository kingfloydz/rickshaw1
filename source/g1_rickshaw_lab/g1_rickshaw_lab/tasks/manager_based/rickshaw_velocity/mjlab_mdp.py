"""Mjlab observation adapters and Rickshaw-specific rewards."""

from __future__ import annotations

from typing import Any

import torch
from mjlab.tasks.velocity import mdp as velocity_mdp

from g1_rickshaw_lab.policy_schema import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_DIM,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
)

from .mdp.observations import (
    assemble_actor_observation,
    assemble_teacher_dynamic_privilege,
)
from .mdp.rewards import (
    HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    HITCH_HEIGHT_RECOVERY_SCALE_M,
    angle_deviation_l2_value,
    hitch_height_recovery_l2_value,
    peak_force_value,
    relative_position_l2_value,
    wheel_slip_l2_value,
)
from .mjlab_events import ensure_mjlab_observation_state, ensure_mjlab_reward_state


def _shape_probe(env: Any, *shape: int) -> torch.Tensor:
    if hasattr(env, "observation_manager"):
        raise RuntimeError("observation state was not initialized by the startup event")
    return torch.empty((env.num_envs, *shape), device=env.device)


def _dynamic_privilege(env: Any) -> torch.Tensor:
    result = assemble_teacher_dynamic_privilege(
        torch.stack((env.rickshaw_speed_s, env.rickshaw_ang_vel_z), dim=-1),
        env.rickshaw_state.pitch,
        env.rickshaw_state.wheel_normal_force,
        env.rickshaw_state.connection_force_w,
        env.path_tangent_w,
        env.path_lateral_w,
        env.path_normal_w,
    )
    if result.shape != (env.num_envs, TEACHER_DYNAMIC_DIM):
        raise RuntimeError(f"teacher dynamic observation is not {TEACHER_DYNAMIC_DIM}-D")
    return result


def _actor_observation(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    command = env.command_manager.get_command("twist")
    action_term = env.action_manager.get_term("joint_pos")
    return assemble_actor_observation(
        robot.data.root_link_lin_vel_b,
        robot.data.root_link_ang_vel_b,
        robot.data.projected_gravity_b,
        command[:, (0, 2)],
        robot.data.joint_pos[:, env.policy_joint_ids],
        action_term.q_ref,
        robot.data.joint_vel[:, env.policy_joint_ids],
        env.action_manager.action,
    )


def current_actor_observation(env: Any) -> torch.Tensor:
    if not hasattr(env, "rickshaw_state"):
        return _shape_probe(env, ACTOR_OBSERVATION_DIM)
    ensure_mjlab_observation_state(env)
    return _actor_observation(env)


def critic_actor_observation(env: Any) -> torch.Tensor:
    if not hasattr(env, "rickshaw_state"):
        return _shape_probe(env, ACTOR_OBSERVATION_DIM)
    ensure_mjlab_observation_state(env)
    return _actor_observation(env)


def teacher_dynamic_observation(env: Any) -> torch.Tensor:
    if not hasattr(env, "rickshaw_state"):
        return _shape_probe(env, TEACHER_DYNAMIC_DIM)
    ensure_mjlab_observation_state(env)
    return _dynamic_privilege(env)


def teacher_static(env: Any) -> torch.Tensor:
    if not hasattr(env, "normalized_teacher_static_domain"):
        return _shape_probe(env, TEACHER_STATIC_DIM)
    return env.normalized_teacher_static_domain


def critic_privileged_state(env: Any) -> torch.Tensor:
    if not hasattr(env, "rickshaw_state"):
        return _shape_probe(env, CRITIC_PRIVILEGED_DIM)
    ensure_mjlab_observation_state(env)
    result = torch.cat(
        (
            teacher_static(env),
            _dynamic_privilege(env),
            env.rickshaw_kinematic_state.forward_acceleration[:, None],
            env.rickshaw_kinematic_state.yaw_angular_acceleration[:, None],
            velocity_mdp.foot_height(env, "foot_height_scan"),
            velocity_mdp.foot_air_time(env, "feet_ground_contact"),
            velocity_mdp.foot_contact(env, "feet_ground_contact"),
            velocity_mdp.foot_contact_forces(env, "feet_ground_contact"),
        ),
        dim=-1,
    )
    if result.shape != (env.num_envs, CRITIC_PRIVILEGED_DIM):
        raise RuntimeError(f"critic privileged observation is not {CRITIC_PRIVILEGED_DIM}-D")
    return result


def track_rickshaw_lin_vel_x(env: Any, command_name: str, std: float) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    command = env.command_manager.get_command(command_name)
    return torch.exp(-torch.square(env.rickshaw_speed_s - command[:, 0]) / std**2)


def track_rickshaw_ang_vel_z(env: Any, command_name: str, std: float) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    command = env.command_manager.get_command(command_name)
    return torch.exp(-torch.square(env.rickshaw_ang_vel_z - command[:, 2]) / std**2)


def rickshaw_forward_acceleration_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return torch.square(env.rickshaw_kinematic_state.forward_acceleration)


def rickshaw_pitch_angular_acceleration_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return torch.square(env.rickshaw_kinematic_state.pitch_angular_acceleration)


def rickshaw_yaw_angular_acceleration_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return torch.square(env.rickshaw_kinematic_state.yaw_angular_acceleration)


def rickshaw_pitch_angular_velocity_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return torch.square(env.rickshaw_kinematic_state.pitch_angular_velocity)


def rickshaw_wheel_slip_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return wheel_slip_l2_value(
        env.rickshaw_state.wheel_longitudinal_slip,
        env.rickshaw_state.wheel_normal_force > 1.0,
    )


def rickshaw_g1_relative_position_l2(env: Any, axle_weight: float = 4.0) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return relative_position_l2_value(
        env.rickshaw_state.relative_position_b,
        env.static_relative_position_b,
        axle_weight=axle_weight,
    )


def rickshaw_g1_relative_yaw_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return angle_deviation_l2_value(env.rickshaw_state.relative_yaw, env.static_relative_yaw)


def rickshaw_absolute_pitch_deviation_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return angle_deviation_l2_value(env.rickshaw_state.pitch, env.static_rickshaw_pitch)


def peak_force(env: Any, soft_limit: float = 10.0, hard_limit: float = 50.0) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return peak_force_value(
        env.rickshaw_state.connection_force_w,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
    )


def hitch_height_recovery_l2(
    env: Any,
    deadband: float = HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    scale: float = HITCH_HEIGHT_RECOVERY_SCALE_M,
) -> torch.Tensor:
    ensure_mjlab_reward_state(env)
    return hitch_height_recovery_l2_value(
        env.rickshaw_state.hitch_height,
        env.hitch_height_target,
        deadband=deadband,
        scale=scale,
    )
