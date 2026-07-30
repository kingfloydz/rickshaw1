"""Mjlab configuration for G1 towing the site-connected rickshaw."""

from __future__ import annotations

import math
import os
from typing import Any, Literal

from g1_rickshaw_lab.assets.g1_dex1 import get_g1_robot_cfg
from g1_rickshaw_lab.assets.rickshaw import get_rickshaw_cfg
from g1_rickshaw_lab.configuration import load_feasibility_envelope
from g1_rickshaw_lab.policy_schema import HISTORY_LENGTH, validate_history_length
from g1_rickshaw_lab.project_paths import CONFIG_ROOT, PROJECT_ROOT

PLAY_SLOPE_ENV = "G1_RICKSHAW_PLAY_SLOPE"

_REWARD_CURRICULUM_SCHEDULES = {
    "s0": (
        (0, 0.02, 0.04),
        (300, 0.10, 0.15),
        (600, 0.25, 0.30),
        (900, 0.45, 0.50),
        (1200, 0.70, 0.75),
        (1500, 1.00, 1.00),
    ),
    "s2": (
        (0, 0.05, 0.10),
        (100, 0.20, 0.25),
        (200, 0.50, 0.50),
        (300, 0.75, 0.75),
        (400, 1.00, 1.00),
    ),
}


def configure_history_length(env_cfg: Any, history_length: int) -> None:
    history_length = validate_history_length(history_length)
    env_cfg.history_length = history_length
    env_cfg.observations["policy_sequence"].history_length = history_length + 1
    env_cfg.observations["teacher_dynamic_sequence"].history_length = history_length + 1


def _play_slope() -> float | None:
    value = os.environ.get(PLAY_SLOPE_ENV)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{PLAY_SLOPE_ENV} must be a slope in radians") from exc


def _runtime_cfg(*, play: bool, terrain_slope: float | None):
    from .mdp.events import DomainRandomizationCfg
    from .mjlab_events import MjlabTaskRuntimeCfg

    envelope = load_feasibility_envelope(CONFIG_ROOT / "feasibility_envelope.yaml")
    calibration = dict(envelope.calibration)
    names = (
        "torso.mass_delta",
        "rickshaw.mass_delta",
        "rolling_resistance.c_rr",
        "terrain.friction",
        "wheel.left_damping",
        "wheel.right_damping",
    )
    ranges = {name: (envelope.ranges[name].minimum, envelope.ranges[name].maximum) for name in names}
    nominal = {
        "torso.mass_delta": 0.0,
        "rickshaw.mass_delta": 0.0,
        "rolling_resistance.c_rr": calibration["rolling_resistance.c_rr_nominal"],
        "terrain.friction": calibration["terrain.friction_nominal"],
        "wheel.left_damping": 0.02,
        "wheel.right_damping": 0.02,
    }
    domain = DomainRandomizationCfg(
        enabled=not play,
        ranges=ranges,
        nominal=nominal,
        calibration=calibration,
    )
    return MjlabTaskRuntimeCfg(
        domain=domain,
        terrain_slope=terrain_slope,
    )


def g1_rickshaw_env_cfg(
    *,
    play: bool = False,
    history_length: int = HISTORY_LENGTH,
    reward_curriculum: Literal["s0", "s2"] | None = "s0",
    terrain_slope: float | None = None,
):
    """Create the nineteen-slope rickshaw velocity task using mjlab 1.5.3 APIs."""

    history_length = validate_history_length(history_length)
    if terrain_slope is None and play:
        terrain_slope = _play_slope()

    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.sensor import (
        ContactMatch,
        ContactSensorCfg,
    )
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    from mjlab.utils.nan_guard import NanGuardCfg
    from mjlab.utils.noise import UniformNoiseCfg

    from . import mjlab_mdp
    from .closed_chain import add_closed_chain_constraints
    from .mdp.observations import ACTOR_OBSERVATION_NOISE_SCALE
    from .mdp.rewards import (
        HITCH_HEIGHT_RECOVERY_DEADBAND_M,
        HITCH_HEIGHT_RECOVERY_SCALE_M,
    )
    from .mjlab_actions import StaticReferenceJointPositionActionCfg
    from .mjlab_commands import RickshawVelocityCommandCfg
    from .mjlab_events import (
        apply_mjlab_rolling_resistance,
        initialize_mjlab_domain,
        initialize_mjlab_task,
        reset_from_mujoco_static,
    )

    cfg = unitree_g1_flat_env_cfg(play=play)
    runtime = _runtime_cfg(play=play, terrain_slope=terrain_slope)
    actor_noise = UniformNoiseCfg(
        n_min=tuple(-value for value in ACTOR_OBSERVATION_NOISE_SCALE),
        n_max=ACTOR_OBSERVATION_NOISE_SCALE,
    )
    observations = {
        "policy_sequence": ObservationGroupCfg(
            terms={
                "observation": ObservationTermCfg(
                    func=mjlab_mdp.current_actor_observation,
                    noise=actor_noise,
                )
            },
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=history_length + 1,
            flatten_history_dim=False,
        ),
        "teacher_dynamic_sequence": ObservationGroupCfg(
            terms={"observation": ObservationTermCfg(func=mjlab_mdp.teacher_dynamic_observation)},
            concatenate_terms=True,
            enable_corruption=False,
            history_length=history_length + 1,
            flatten_history_dim=False,
        ),
        "teacher_static": ObservationGroupCfg(
            terms={"static": ObservationTermCfg(func=mjlab_mdp.teacher_static)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms={"privileged": ObservationTermCfg(func=mjlab_mdp.critic_privileged_state)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic_policy": ObservationGroupCfg(
            terms={"current": ObservationTermCfg(func=mjlab_mdp.critic_actor_observation)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    actions = {
        "joint_pos": StaticReferenceJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            preserve_order=True,
        )
    }
    events = {
        "initialize_task": EventTermCfg(func=initialize_mjlab_task, mode="startup", params={"cfg": runtime}),
        "initialize_domain": EventTermCfg(func=initialize_mjlab_domain, mode="startup", params={"cfg": runtime}),
        "mujoco_static_reset": EventTermCfg(func=reset_from_mujoco_static, mode="reset"),
        "rolling_resistance": EventTermCfg(func=apply_mjlab_rolling_resistance, mode="step"),
    }
    feet_ground_sensor_name = "feet_ground_contact"
    foot_height_sensor_name = "foot_height_scan"
    self_collision_sensor_name = "self_collision"
    rewards = cfg.rewards
    rewards["track_linear_velocity"] = RewardTermCfg(
        func=mjlab_mdp.track_rickshaw_lin_vel_x,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
    )
    rewards["track_angular_velocity"] = RewardTermCfg(
        func=mjlab_mdp.track_rickshaw_ang_vel_z,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
    )
    rewards["upright"].weight = 0.1
    rewards["upright"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("torso_link",))

    pose = rewards["pose"]
    pose.weight = 0.5
    pose.params["std_standing"] = {
        pattern: 0.05
        for pattern in (
            ".*hip_pitch.*",
            ".*hip_roll.*",
            ".*hip_yaw.*",
            ".*knee.*",
            ".*ankle_pitch.*",
            ".*ankle_roll.*",
            ".*waist_yaw.*",
            ".*waist_roll.*",
            ".*waist_pitch.*",
            ".*shoulder_pitch.*",
            ".*shoulder_roll.*",
            ".*shoulder_yaw.*",
            ".*elbow.*",
            ".*wrist.*",
        )
    }
    upper_body_patterns = (
        ".*waist_pitch.*",
        ".*shoulder_pitch.*",
        ".*shoulder_roll.*",
        ".*shoulder_yaw.*",
        ".*elbow.*",
        ".*wrist.*",
    )
    for standard_deviations in (
        pose.params["std_standing"],
        pose.params["std_walking"],
        pose.params["std_running"],
    ):
        for pattern in upper_body_patterns:
            standard_deviations[pattern] *= math.sqrt(3.0)
    rewards["body_ang_vel"].weight = -0.05
    rewards["angular_momentum"].weight = -0.02
    rewards["foot_swing_height"].params["target_height"] = 0.08

    rewards.update(
        {
            "arm_joint_velocity_l2": RewardTermCfg(
                func=velocity_mdp.joint_vel_l2,
                weight=-0.0015,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(r".*_(shoulder|elbow|wrist)_.*",),
                    )
                },
            ),
            "rickshaw_forward_acceleration_l2": RewardTermCfg(
                func=mjlab_mdp.rickshaw_forward_acceleration_l2,
                weight=-0.05,
            ),
            "rickshaw_pitch_angular_acceleration_l2": RewardTermCfg(
                func=mjlab_mdp.rickshaw_pitch_angular_acceleration_l2,
                weight=-0.01,
            ),
            "rickshaw_yaw_angular_acceleration_l2": RewardTermCfg(
                func=mjlab_mdp.rickshaw_yaw_angular_acceleration_l2,
                weight=-0.01,
            ),
            "rickshaw_pitch_angular_velocity_l2": RewardTermCfg(
                func=mjlab_mdp.rickshaw_pitch_angular_velocity_l2,
                weight=-1.0,
            ),
            "rickshaw_wheel_slip_l2": RewardTermCfg(
                func=mjlab_mdp.rickshaw_wheel_slip_l2,
                weight=-0.1,
            ),
            "rickshaw_g1_relative_position_l2": RewardTermCfg(
                func=mjlab_mdp.rickshaw_g1_relative_position_l2,
                weight=-4.0,
                params={"axle_weight": 4.0},
            ),
            "rickshaw_g1_relative_yaw_l2": RewardTermCfg(
                func=mjlab_mdp.rickshaw_g1_relative_yaw_l2,
                weight=-0.6,
            ),
            "rickshaw_absolute_pitch_deviation_l2": RewardTermCfg(
                func=mjlab_mdp.rickshaw_absolute_pitch_deviation_l2,
                weight=-0.5,
            ),
            "peak_force": RewardTermCfg(
                func=mjlab_mdp.peak_force,
                weight=-3.0,
                params={"soft_limit": 10.0, "hard_limit": 50.0},
            ),
            "hitch_height_recovery_l2": RewardTermCfg(
                func=mjlab_mdp.hitch_height_recovery_l2,
                weight=-0.25,
                params={
                    "deadband": HITCH_HEIGHT_RECOVERY_DEADBAND_M,
                    "scale": HITCH_HEIGHT_RECOVERY_SCALE_M,
                },
            ),
        }
    )
    ramped_dynamics_penalties = (
        "rickshaw_forward_acceleration_l2",
        "rickshaw_pitch_angular_acceleration_l2",
        "rickshaw_yaw_angular_acceleration_l2",
        "rickshaw_pitch_angular_velocity_l2",
        "rickshaw_absolute_pitch_deviation_l2",
        "peak_force",
    )
    ramped_relative_pose_penalties = (
        "rickshaw_g1_relative_position_l2",
        "rickshaw_g1_relative_yaw_l2",
    )
    reward_weight_curricula = {}
    if not play and reward_curriculum is not None:
        schedule = _REWARD_CURRICULUM_SCHEDULES[reward_curriculum]
        for names, multiplier_index in (
            (ramped_dynamics_penalties, 1),
            (ramped_relative_pose_penalties, 2),
        ):
            for name in names:
                reward_weight_curricula[f"{name}_weight"] = CurriculumTermCfg(
                    func=velocity_mdp.reward_curriculum,
                    params={
                        "reward_name": name,
                        "stages": [
                            {
                                "step": stage[0] * 24,
                                "weight": rewards[name].weight * stage[multiplier_index],
                            }
                            for stage in schedule
                        ],
                    },
                )
    commands = {
        "twist": RickshawVelocityCommandCfg(
            entity_name="rickshaw",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            rel_forward_envs=0.2,
            heading_command=False,
            debug_vis=True,
            ranges=RickshawVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.5, 2.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.7, 0.7),
            ),
        )
    }
    commands["twist"].viz.z_offset = 1.15
    terminations = {
        "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=velocity_mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }
    base_sensors = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
    wheels = ContactSensorCfg(
        name="wheel_contacts",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(left_wheel_link|right_wheel_link)$",
            entity="rickshaw",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.entities = {"robot": get_g1_robot_cfg(), "rickshaw": get_rickshaw_cfg()}
    cfg.scene.sensors = (
        base_sensors[feet_ground_sensor_name],
        base_sensors[foot_height_sensor_name],
        base_sensors[self_collision_sensor_name],
        wheels,
    )
    cfg.scene.num_envs = 1 if play else 8192
    cfg.scene.env_spacing = 6.0
    cfg.scene.spec_fn = add_closed_chain_constraints
    cfg.observations = observations
    cfg.actions = actions
    cfg.commands = commands
    cfg.events = events
    cfg.rewards = rewards
    cfg.terminations = terminations
    cfg.curriculum = reward_weight_curricula
    cfg.metrics = {"mean_action_acc": MetricsTermCfg(func=velocity_mdp.mean_action_acc)}
    cfg.sim.nconmax = None
    cfg.sim.njmax = 600
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nan_guard = NanGuardCfg(
        enabled=not play,
        buffer_size=100,
        output_dir=str(PROJECT_ROOT / "outputs" / "nan_dumps"),
        max_envs_to_dump=5,
    )
    cfg.sim.mujoco.timestep = 0.005
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 20
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.history_length = history_length
    cfg.domain_randomization = runtime.domain
    if play:
        cfg.episode_length_s = int(1e9)
    cfg.terrain_slope = terrain_slope
    return cfg
