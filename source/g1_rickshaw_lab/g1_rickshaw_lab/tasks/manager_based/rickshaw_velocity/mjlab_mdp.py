"""Mjlab observation adapters and Rickshaw-specific rewards."""

from __future__ import annotations

from typing import Any

import torch

from g1_rickshaw_lab.policy_schema import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
)

from .mdp.observations import (
    ACTOR_OBSERVATION_NOISE_SCALE,
    assemble_actor_observation,
    gait_phase_observation,
)
from .mdp.rewards import (
    HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    HITCH_HEIGHT_RECOVERY_SCALE_M,
    hitch_height_exp_value,
    hitch_height_recovery_l2_value,
)
from .mjlab_events import ensure_mjlab_physical_state


def _shape_probe(env: Any, *shape: int) -> torch.Tensor:
    if hasattr(env, "observation_manager"):
        raise RuntimeError("observation state was not initialized by the startup event")
    return torch.empty((env.num_envs, *shape), device=env.device)


def _dynamic_privilege(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]

    def project(vector: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                torch.sum(vector * env.path_tangent_w, dim=-1),
                torch.sum(vector * env.path_lateral_w, dim=-1),
                torch.sum(vector * env.path_normal_w, dim=-1),
            ),
            dim=-1,
        )

    basis = torch.stack((env.path_tangent_w, env.path_lateral_w, env.path_normal_w), dim=1)
    wrench = env.rickshaw_state.connection_truth_wrench_w
    force_sln = torch.einsum("nsw,ncw->nsc", wrench[..., :3], basis)
    torque_sln = torch.einsum("nsw,ncw->nsc", wrench[..., 3:], basis)
    result = torch.cat(
        (
            project(robot.data.root_link_lin_vel_w),
            project(cart.data.root_link_lin_vel_w),
            env.rickshaw_state.pitch[:, None],
            env.rickshaw_state.wheel_normal_force,
            torch.cat((force_sln, torque_sln), dim=-1).reshape(env.num_envs, -1),
        ),
        dim=-1,
    )
    if result.shape != (env.num_envs, TEACHER_DYNAMIC_DIM):
        raise RuntimeError("teacher dynamic observation is not 21-D")
    return result


def _update_observation_state(env: Any) -> None:
    ensure_mjlab_physical_state(env)
    step = int(env.common_step_counter)
    if env._mjlab_observation_state_step == step:
        return
    env._mjlab_observation_state_step = step
    robot = env.scene["robot"]
    command = env.command_manager.get_command("twist")
    current = assemble_actor_observation(
        robot.data.root_link_ang_vel_b,
        robot.data.projected_gravity_b,
        command[:, 0],
        env.path_state.lateral_error,
        env.path_state.heading_error,
        robot.data.joint_pos[:, env.policy_joint_ids],
        env.action_state.q_ref,
        robot.data.joint_vel[:, env.policy_joint_ids],
        env.action_state.target,
        gait_phase_observation(env.episode_length_buf * env.step_dt),
    )
    if env.cfg.observation_noise_enabled:
        noise = torch.tensor(ACTOR_OBSERVATION_NOISE_SCALE, device=env.device)
        current = current + torch.empty_like(current).uniform_(-1.0, 1.0) * noise
    env.observation_history_state.advance(current)
    env.teacher_dynamic_history_state.advance(_dynamic_privilege(env))


def current_actor_observation(env: Any) -> torch.Tensor:
    if not hasattr(env, "observation_history_state"):
        return _shape_probe(env, ACTOR_OBSERVATION_DIM)
    _update_observation_state(env)
    return env.observation_history_state.current


def actor_observation_history(env: Any, history_length: int = HISTORY_LENGTH) -> torch.Tensor:
    if not hasattr(env, "observation_history_state"):
        return _shape_probe(env, history_length, ACTOR_OBSERVATION_DIM)
    _update_observation_state(env)
    history = env.observation_history_state.history
    if history is None:
        raise RuntimeError("actor history is disabled")
    return history


def teacher_dynamic_history(env: Any, history_length: int = HISTORY_LENGTH) -> torch.Tensor:
    if not hasattr(env, "teacher_dynamic_history_state"):
        return _shape_probe(env, history_length, TEACHER_DYNAMIC_DIM)
    _update_observation_state(env)
    history = env.teacher_dynamic_history_state.history
    if history is None:
        raise RuntimeError("teacher dynamic history is disabled")
    return history


def teacher_static(env: Any) -> torch.Tensor:
    if not hasattr(env, "normalized_teacher_static_domain"):
        return _shape_probe(env, TEACHER_STATIC_DIM)
    return env.normalized_teacher_static_domain


def critic_privileged_state(env: Any) -> torch.Tensor:
    if not hasattr(env, "teacher_dynamic_history_state"):
        return _shape_probe(env, CRITIC_PRIVILEGED_DIM)
    _update_observation_state(env)
    result = torch.cat(
        (
            teacher_static(env),
            env.teacher_dynamic_history_state.current,
            env.rickshaw_state.connection_residual[:, None],
            env.stability_state.zmp_margin[:, None],
            env.analytic_force_state.a_s[:, None],
        ),
        dim=-1,
    )
    if result.shape != (env.num_envs, CRITIC_PRIVILEGED_DIM):
        raise RuntimeError("critic privileged observation is not 33-D")
    return result


def hitch_height_exp(env: Any, std: float) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return hitch_height_exp_value(
        env.rickshaw_state.hitch_height,
        env.hitch_height_target,
        env.rickshaw_state.two_wheel_contact,
        std=std,
    )


def hitch_height_recovery_l2(
    env: Any,
    deadband: float = HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    scale: float = HITCH_HEIGHT_RECOVERY_SCALE_M,
) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return hitch_height_recovery_l2_value(
        env.rickshaw_state.hitch_height,
        env.hitch_height_target,
        deadband=deadband,
        scale=scale,
    )


__all__ = [
    "actor_observation_history",
    "critic_privileged_state",
    "current_actor_observation",
    "hitch_height_exp",
    "hitch_height_recovery_l2",
    "teacher_dynamic_history",
    "teacher_static",
]
