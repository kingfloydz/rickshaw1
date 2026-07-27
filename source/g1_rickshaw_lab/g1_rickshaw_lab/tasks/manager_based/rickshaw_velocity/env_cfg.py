"""Mjlab configuration for G1 towing the site-connected rickshaw."""

from __future__ import annotations

import math

from g1_rickshaw_lab.assets import get_g1_robot_cfg, get_rickshaw_cfg
from g1_rickshaw_lab.configuration import G1_JOINT_ORDER, load_feasibility_envelope
from g1_rickshaw_lab.policy_schema import HISTORY_LENGTH
from g1_rickshaw_lab.project_paths import CONFIG_ROOT, PROJECT_ROOT

from .mdp.mimic import LEG_TORSO_JOINT_COUNT

MIMIC_MOTION_PATH = PROJECT_ROOT / "hmr4d_results_straight_g1.pkl"


def enable_mimic(env_cfg):
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from . import mjlab_mdp

    command = env_cfg.commands["twist"]
    command.mimic = True
    command.mimic_motion_path = str(MIMIC_MOTION_PATH)
    mimic_joints = SceneEntityCfg(
        "robot",
        joint_names=G1_JOINT_ORDER[:LEG_TORSO_JOINT_COUNT],
        preserve_order=True,
    )
    env_cfg.rewards["mimic_joint_position"] = RewardTermCfg(
        func=mjlab_mdp.mimic_joint_position_exp,
        weight=2.0,
        params={
            "command_name": "twist",
            "std": 0.3,
            "asset_cfg": mimic_joints,
        },
    )
    env_cfg.rewards["mimic_joint_velocity"] = RewardTermCfg(
        func=mjlab_mdp.mimic_joint_velocity_exp,
        weight=30.0,
        params={
            "command_name": "twist",
            "std": 1.0,
            "asset_cfg": mimic_joints,
        },
    )
    env_cfg.mimic = True
    return env_cfg


def _runtime_cfg(*, play: bool, history_length: int):
    from .mdp.dynamics import AnalyticForceCfg, FAT2Cfg, SupportPolygonCfg, ZMPCfg
    from .mdp.events import DomainRandomizationCfg
    from .mjlab_events import MjlabTaskRuntimeCfg
    from .task_spec import RickshawPoseTargetCfg

    envelope = load_feasibility_envelope(CONFIG_ROOT / "feasibility_envelope.yaml")
    calibration = dict(envelope.calibration)
    names = (
        "torso.mass_delta",
        "payload.mass",
        "payload.com.x",
        "payload.com.y",
        "payload.com.z",
        "rolling_resistance.c_rr",
        "terrain.friction",
        "wheel.left_damping",
        "wheel.right_damping",
    )
    ranges = {name: (envelope.ranges[name].minimum, envelope.ranges[name].maximum) for name in names}
    nominal = {
        "torso.mass_delta": 0.0,
        "payload.mass": 0.0,
        "payload.com.x": 0.5 * sum(ranges["payload.com.x"]),
        "payload.com.y": 0.0,
        "payload.com.z": 0.5 * sum(ranges["payload.com.z"]),
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
        rickshaw_pose=RickshawPoseTargetCfg(
            hitch_height_tolerance=calibration["rickshaw_pose.hitch_height_tolerance"],
            hitch_vertical_speed_tolerance=calibration["rickshaw_pose.hitch_vertical_speed_tolerance"],
        ),
        analytic_force=AnalyticForceCfg(minimum_wheel_normal_force=calibration["safety.minimum_wheel_normal_force"]),
        fat2=FAT2Cfg(
            robot_mass=calibration["fat.robot_mass"],
            com_radius=calibration["fat.com_radius"],
            com_radius_bounds=tuple(calibration["fat.com_radius_bounds"]),
            theta_max=calibration["safety.theta_max"],
            force_consistency_relative_tolerance=calibration["fat.force_consistency_relative_tolerance"],
            force_consistency_absolute_floor_n=calibration["fat.force_consistency_absolute_floor_n"],
        ),
        support=SupportPolygonCfg(
            foot_half_length=calibration["support.foot_half_length"],
            foot_half_width=calibration["support.foot_half_width"],
            foot_center_offset_x=calibration["support.foot_center_offset_x"],
        ),
        zmp=ZMPCfg(min_ground_reaction=calibration["safety.min_ground_reaction"]),
        history_length=history_length,
        play=play,
    )


def g1_rickshaw_env_cfg(
    *,
    play: bool = False,
    history_length: int = HISTORY_LENGTH,
    mimic: bool = False,
):
    """Create the nineteen-slope rickshaw velocity task using mjlab 1.5.3 APIs."""

    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sensor import (
        ContactMatch,
        ContactSensorCfg,
        ObjRef,
        RingPatternCfg,
        TerrainHeightSensorCfg,
    )
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.utils.nan_guard import NanGuardCfg
    from mjlab.viewer import ViewerConfig

    from . import mjlab_mdp
    from .closed_chain import add_closed_chain_constraints
    from .mdp.rewards import (
        HITCH_HEIGHT_ERROR_SCALE_M,
        HITCH_HEIGHT_RECOVERY_DEADBAND_M,
        HITCH_HEIGHT_RECOVERY_SCALE_M,
    )
    from .mjlab_actions import StaticReferenceJointPositionActionCfg
    from .mjlab_commands import RickshawVelocityCommandCfg
    from .mjlab_events import (
        advance_mjlab_policy_state,
        initialize_mjlab_domain,
        initialize_mjlab_task,
        reset_from_mujoco_static,
    )

    runtime = _runtime_cfg(play=play, history_length=history_length)
    observations = {
        "policy": ObservationGroupCfg(
            terms={"current": ObservationTermCfg(func=mjlab_mdp.current_actor_observation)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "history": ObservationGroupCfg(
            terms={
                "history": ObservationTermCfg(
                    func=mjlab_mdp.actor_observation_history,
                    params={"history_length": history_length},
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "teacher_dynamic_history": ObservationGroupCfg(
            terms={
                "history": ObservationTermCfg(
                    func=mjlab_mdp.teacher_dynamic_history,
                    params={"history_length": history_length},
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
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
        "policy_state": EventTermCfg(func=advance_mjlab_policy_state, mode="step", params={"cfg": runtime}),
    }
    feet_ground_sensor_name = "feet_ground_contact"
    foot_height_sensor_name = "foot_height_scan"
    self_collision_sensor_name = "self_collision"
    foot_site_names = ("left_foot", "right_foot")
    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mjlab_mdp.track_rickshaw_lin_vel_x,
            weight=2.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.25),
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mjlab_mdp.track_rickshaw_ang_vel_z,
            weight=2.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.5),
            },
        ),
        "upright": RewardTermCfg(
            func=velocity_mdp.upright,
            weight=0.1,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            },
        ),
        "pose": RewardTermCfg(
            func=velocity_mdp.variable_posture,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(r".*",),
                ),
                "command_name": "twist",
                "std_standing": {
                    ".*hip_pitch.*": 0.05,
                    ".*hip_roll.*": 0.05,
                    ".*hip_yaw.*": 0.05,
                    ".*knee.*": 0.05,
                    ".*ankle_pitch.*": 0.05,
                    ".*ankle_roll.*": 0.05,
                    ".*waist_yaw.*": 0.05,
                    ".*waist_roll.*": 0.05,
                    ".*waist_pitch.*": 0.05 * math.sqrt(3.0),
                    ".*shoulder_pitch.*": 0.05 * math.sqrt(3.0),
                    ".*shoulder_roll.*": 0.05 * math.sqrt(3.0),
                    ".*shoulder_yaw.*": 0.05 * math.sqrt(3.0),
                    ".*elbow.*": 0.05 * math.sqrt(3.0),
                    ".*wrist.*": 0.05 * math.sqrt(3.0),
                },
                "std_walking": {
                    ".*hip_pitch.*": 0.3,
                    ".*hip_roll.*": 0.15,
                    ".*hip_yaw.*": 0.15,
                    ".*knee.*": 0.35,
                    ".*ankle_pitch.*": 0.25,
                    ".*ankle_roll.*": 0.1,
                    ".*waist_yaw.*": 0.2,
                    ".*waist_roll.*": 0.08,
                    ".*waist_pitch.*": 0.1 * math.sqrt(3.0),
                    ".*shoulder_pitch.*": 0.15 * math.sqrt(3.0),
                    ".*shoulder_roll.*": 0.15 * math.sqrt(3.0),
                    ".*shoulder_yaw.*": 0.1 * math.sqrt(3.0),
                    ".*elbow.*": 0.15 * math.sqrt(3.0),
                    ".*wrist.*": 0.3 * math.sqrt(3.0),
                },
                "std_running": {
                    ".*hip_pitch.*": 0.5,
                    ".*hip_roll.*": 0.2,
                    ".*hip_yaw.*": 0.2,
                    ".*knee.*": 0.6,
                    ".*ankle_pitch.*": 0.35,
                    ".*ankle_roll.*": 0.15,
                    ".*waist_yaw.*": 0.3,
                    ".*waist_roll.*": 0.08,
                    ".*waist_pitch.*": 0.2 * math.sqrt(3.0),
                    ".*shoulder_pitch.*": 0.5 * math.sqrt(3.0),
                    ".*shoulder_roll.*": 0.2 * math.sqrt(3.0),
                    ".*shoulder_yaw.*": 0.15 * math.sqrt(3.0),
                    ".*elbow.*": 0.35 * math.sqrt(3.0),
                    ".*wrist.*": 0.3 * math.sqrt(3.0),
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=velocity_mdp.body_angular_velocity_penalty,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
        ),
        "angular_momentum": RewardTermCfg(
            func=velocity_mdp.angular_momentum_penalty,
            weight=-0.02,
            params={"sensor_name": "robot/root_angmom"},
        ),
        "dof_pos_limits": RewardTermCfg(func=velocity_mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=velocity_mdp.action_rate_l2, weight=-0.1),
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
            weight=-1,
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
        "air_time": RewardTermCfg(
            func=velocity_mdp.feet_air_time,
            weight=0.0,
            params={
                "sensor_name": feet_ground_sensor_name,
                "threshold_min": 0.05,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.5,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=velocity_mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": 0.1,
                "height_sensor_name": foot_height_sensor_name,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=velocity_mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_name": feet_ground_sensor_name,
                "height_sensor_name": foot_height_sensor_name,
                "target_height": 0.08,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=velocity_mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": feet_ground_sensor_name,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=velocity_mdp.soft_landing,
            weight=-1.0e-5,
            params={
                "sensor_name": feet_ground_sensor_name,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "self_collisions": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": self_collision_sensor_name, "force_threshold": 10.0},
        ),
        "hitch_height_exp": RewardTermCfg(
            func=mjlab_mdp.hitch_height_exp,
            weight=0.5,
            params={"std": HITCH_HEIGHT_ERROR_SCALE_M},
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
    ramped_rickshaw_penalties = (
        "rickshaw_forward_acceleration_l2",
        "rickshaw_pitch_angular_acceleration_l2",
        "rickshaw_yaw_angular_acceleration_l2",
        "rickshaw_pitch_angular_velocity_l2",
        "rickshaw_wheel_slip_l2",
        "rickshaw_g1_relative_position_l2",
        "rickshaw_g1_relative_yaw_l2",
        "rickshaw_absolute_pitch_deviation_l2",
        "peak_force",
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
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.5, 0.5),
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
    feet = ContactSensorCfg(
        name=feet_ground_sensor_name,
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    foot_height = TerrainHeightSensorCfg(
        name=foot_height_sensor_name,
        frame=tuple(ObjRef(type="site", name=name, entity="robot") for name in foot_site_names),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    self_collision = ContactSensorCfg(
        name=self_collision_sensor_name,
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
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
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_robot_cfg(), "rickshaw": get_rickshaw_cfg()},
            sensors=(feet, foot_height, self_collision, wheels),
            num_envs=1 if play else 8192,
            env_spacing=6.0,
            extent=2.0,
            spec_fn=add_closed_chain_constraints,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={
            "command_vel": CurriculumTermCfg(
                func=velocity_mdp.commands_vel,
                params={
                    "command_name": "twist",
                    "velocity_stages": [
                        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                        {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
                        {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
                    ],
                },
            ),
            "rickshaw_penalty_weights": CurriculumTermCfg(
                func=mjlab_mdp.SteppedRewardWeightCurriculum,
                params={
                    "reward_names": ramped_rickshaw_penalties,
                    "interval_steps": 200 * 24,
                    "duration_steps": 1200 * 24,
                },
            ),
        },
        metrics={"mean_action_acc": MetricsTermCfg(func=velocity_mdp.mean_action_acc)},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
            elevation=-5.0,
            # The towing path is the local +x axis, so +y gives a true side view.
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=None,
            njmax=600,
            contact_sensor_maxmatch=64,
            nan_guard=NanGuardCfg(
                enabled=not play,
                buffer_size=100,
                output_dir=str(PROJECT_ROOT / "outputs" / "nan_dumps"),
                max_envs_to_dump=5,
            ),
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
    cfg.history_length = history_length
    cfg.mimic = False
    cfg.observation_noise_enabled = not play
    cfg.domain_randomization = runtime.domain
    cfg.policy_update = runtime
    if mimic:
        enable_mimic(cfg)
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.curriculum = {}
        twist_cmd = cfg.commands["twist"]
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)
    return cfg


def G1RickshawSlopesEnvCfg():
    return g1_rickshaw_env_cfg(play=False)


def G1RickshawSlopesPlayEnvCfg():
    return g1_rickshaw_env_cfg(play=True)


__all__ = [
    "G1RickshawSlopesEnvCfg",
    "G1RickshawSlopesPlayEnvCfg",
    "MIMIC_MOTION_PATH",
    "enable_mimic",
    "g1_rickshaw_env_cfg",
]
