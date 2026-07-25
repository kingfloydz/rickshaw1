#!/usr/bin/env python3
"""Produce fixed-seed flat-ground policy diagnostics in Mjlab."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from _mjlab_wrappers import (
    add_mjlab_sources_to_path,
    add_project_source_to_path,
    load_mjlab_configs,
    require_existing_file,
)

add_project_source_to_path()

from g1_rickshaw_lab.policy_evaluation import (  # noqa: E402
    COMMAND_PHASE_LABELS,
    CROSS_CASE_LABELS,
    FORMAL_EVALUATION_COMMAND_PROTOCOL,
    FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
    METRIC_DEFINITIONS,
    POLICY_DIAGNOSTIC_SCHEMA_VERSION,
    PolicyEvaluationAccumulator,
    command_phase_labels,
    connection_wrench_channels,
    evaluate_s2_return_floor,
    validate_s1_baseline_diagnostic_report,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    CHECKPOINT_STAGE_KEY,
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
    normalize_rsl_rl_runner_configuration,
    require_pinned_rsl_rl,
)
from g1_rickshaw_lab.artifact_io import (  # noqa: E402
    utc_timestamp,
    write_json_atomic,
)


DEFAULT_TASK = "Mjlab-G1-Rickshaw-Flat-Student"
SUPPORTED_STAGES = {
    "s0_teacher",
    "s1_context_distillation",
    "s2_student_ppo",
}
EVALUATION_STAGE = "training"


class PolicyHandle:
    """Uniform distribution interface over native RSL and S1 actors."""

    def __init__(self, actor: Any, *, kind: str) -> None:
        if kind not in {"standalone_student", "rsl_student", "rsl_teacher"}:
            raise ValueError(f"unknown policy handle kind {kind!r}")
        self.actor = actor
        self.kind = kind

    def distribution(self, observation: Any):
        if self.kind == "standalone_student":
            context = self.actor.encode(observation["history"])
            policy = self.actor.actor
        else:
            context = self.actor.encode(observation)
            policy = self.actor.policy
        return policy.distribution(observation["policy"], context)


def _parser() -> argparse.ArgumentParser:
    add_mjlab_sources_to_path()
    require_pinned_rsl_rl()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument(
        "--s1-baseline-report",
        default=None,
        help=(
            "Optional S1 fixed-seed TRAINING report for an S2 return comparison."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    parser.add_argument("--max-policy-steps-per-seed", type=int, default=6000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, stage: str) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if not args.seeds:
        raise ValueError("fixed seeds must be non-empty")
    quota_divisor = len(args.seeds) * len(CROSS_CASE_LABELS)
    if (
        args.episodes <= 0
        or args.episodes % quota_divisor != 0
        or args.max_policy_steps_per_seed <= 0
    ):
        raise ValueError(
            "diagnostics require a positive episode quota "
            f"divisible by seeds={quota_divisor}, and a positive step limit"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("fixed seeds must be unique")
    if args.s1_baseline_report is not None and stage != "s2_student_ppo":
        raise ValueError("--s1-baseline-report applies only to S2 diagnostics")


def _configure_evaluation(env_cfg: Any) -> None:
    env_cfg.curriculum = {}
    env_cfg.domain_randomization.enabled = False


def _apply_evaluation_command_protocol(base_env: Any, active: Any) -> None:
    """Drive every counted episode through standing and moving commands."""

    import torch

    if active.dtype != torch.bool or active.shape != (base_env.num_envs,):
        raise ValueError("active evaluation mask must be boolean with shape [num_envs]")
    if not torch.any(active):
        return
    elapsed_s = base_env.episode_length_buf.to(dtype=torch.float32) * float(
        base_env.step_dt
    )
    # Hold zero briefly, accelerate to and cruise at 1 m/s, then brake to a
    # final standing interval before the 20 s timeout.
    moving = (elapsed_s >= 1.0) & (elapsed_s < 10.0)
    command_term = base_env.command_manager.get_term("twist")
    target = moving.to(dtype=command_term.vel_command_b.dtype)
    command_term.vel_command_b[active] = 0.0
    command_term.vel_command_b[active, 0] = target[active]
    command_term.vel_command_w[active] = command_term.vel_command_b[active]
    command_term.is_standing_env[active] = False
    command_term.is_heading_env[active] = False
    command_term.is_world_env[active] = False
    command_term.is_forward_env[active] = True
    command_term.time_left[active] = 21.0


def _episode_fell(*, timed_out: bool, causes: list[str]) -> bool:
    """Classify safety termination as a fall even on the nominal timeout step."""

    return any(cause != "time_out" for cause in causes) or not timed_out


def _load_policy(
    env: Any,
    checkpoint_path: Path,
    checkpoint: Any,
    stage: str,
    device: str,
    task: str,
) -> tuple[PolicyHandle, list[Any]]:
    keepalive: list[Any] = []
    latent_dim = int(
        checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"]["latent_dim"]
    )
    history_length = int(
        checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"]["history_length"]
    )
    if stage == "s1_context_distillation":
        from g1_rickshaw_lab.rl import G1RickshawStudentActor

        state = checkpoint["model_state_dict"]
        model = G1RickshawStudentActor(latent_dim, history_length).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        keepalive.append(model)
        return PolicyHandle(model, kind="standalone_student"), keepalive

    from mjlab.rl import MjlabOnPolicyRunner
    registry_key = "rsl_rl_cfg_entry_point" if stage == "s0_teacher" else "rsl_rl_student_cfg_entry_point"
    agent_cfg = _load_rsl_runner_cfg(task, registry_key, device, latent_dim, history_length)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=device)
    runner.load(
        os.fspath(checkpoint_path),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        strict=True,
    )
    runner.alg.actor.eval()
    keepalive.append(runner)
    kind = "rsl_teacher" if stage == "s0_teacher" else "rsl_student"
    return PolicyHandle(runner.alg.actor, kind=kind), keepalive


def _load_teacher_policy(
    env: Any, checkpoint_path: Path, device: str, task: str
) -> tuple[PolicyHandle, list[Any]]:
    from mjlab.rl import MjlabOnPolicyRunner
    checkpoint = load_stage_checkpoint(
        checkpoint_path,
        expected_stage="s0_teacher",
    )
    latent_dim = int(
        checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"]["latent_dim"]
    )
    history_length = int(
        checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"]["history_length"]
    )
    agent_cfg = _load_rsl_runner_cfg(
        task, "rsl_rl_cfg_entry_point", device, latent_dim, history_length
    )
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=device)
    runner.load(
        os.fspath(checkpoint_path),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        strict=True,
    )
    runner.alg.actor.eval()
    return PolicyHandle(runner.alg.actor, kind="rsl_teacher"), [runner]


def _load_rsl_runner_cfg(
    task: str,
    registry_key: str,
    device: str,
    latent_dim: int,
    history_length: int,
):
    """Load the fixed RSL-RL 5 runner configuration."""

    del task, device
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents.rsl_rl_cfg import (
        g1_rickshaw_student_ppo_runner_cfg,
        g1_rickshaw_teacher_ppo_runner_cfg,
    )

    agent_cfg = (
        g1_rickshaw_teacher_ppo_runner_cfg()
        if registry_key == "rsl_rl_cfg_entry_point"
        else g1_rickshaw_student_ppo_runner_cfg()
    )
    agent_cfg.actor.latent_dim = latent_dim
    agent_cfg.actor.history_length = history_length
    return normalize_rsl_rl_runner_configuration(agent_cfg)


def _sample_metrics(base_env: Any, teacher_kl: Any | None) -> dict[str, Any]:  # noqa: C901
    import torch
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.task_spec import (
        target_pitch_from_hitch_height,
    )

    robot = base_env.scene["robot"]
    state = base_env.rickshaw_state
    stability = base_env.stability_state
    analytic = base_env.analytic_force_state
    actual_speed = base_env.rickshaw_speed_s
    speed_command = base_env.command_manager.get_command("twist")[:, 0]
    speed_error = actual_speed - speed_command
    overspeed_margin = float(
        base_env.runtime_cfg.domain.calibration["safety.overspeed_margin"]
    )
    overspeed = actual_speed > speed_command + overspeed_margin
    pitch_error = state.pitch - target_pitch_from_hitch_height(
        float(base_env.hitch_height_target), base_env.rickshaw_pose_cfg
    )
    hitch_error = state.hitch_height - float(base_env.hitch_height_target)

    contact_sensor = base_env.scene["feet_ground_contact"]
    foot_contact = contact_sensor.data.found > 0
    foot_velocity = robot.data.body_link_lin_vel_w[:, base_env.foot_body_ids]
    foot_s = torch.sum(foot_velocity * base_env.path_tangent_w[:, None, :], dim=-1)
    foot_y = torch.sum(foot_velocity * base_env.path_lateral_w[:, None, :], dim=-1)
    foot_slip = torch.sum(torch.sqrt(foot_s.square() + foot_y.square()) * foot_contact, dim=-1)

    dt = float(base_env.step_dt)
    action = base_env.action_state
    action_rate = torch.sqrt(torch.mean(((action.target - action.prev_target) / dt).square(), dim=-1))
    action_jerk = torch.sqrt(
        torch.mean(((action.target - 2.0 * action.prev_target + action.prev_prev_target) / (dt * dt)).square(), dim=-1)
    )

    ids = base_env.policy_joint_ids
    torque = robot.data.actuator_force
    velocity = robot.data.joint_vel[:, ids]
    from g1_rickshaw_lab.g1_motor_defaults import G1_JOINT_EFFORT_LIMITS

    effort = torch.tensor(G1_JOINT_EFFORT_LIMITS, device=base_env.device)
    torque_margin = 1.0 - torch.abs(torque) / effort
    leg_margin = torch.amin(torque_margin[:, :12], dim=-1)
    arm_margin = torch.amin(torque_margin[:, 15:], dim=-1)
    power = torch.sum(torch.abs(torque * velocity), dim=-1)

    wrench = state.connection_wrench_w
    connection_channels = connection_wrench_channels(wrench)
    connection_force = connection_channels["force"]
    connection_torque = connection_channels["torque"]
    force_asymmetry = connection_channels["force_asymmetry"]
    torque_asymmetry = connection_channels["torque_asymmetry"]
    # The adapter stores cart-on-robot reaction; analytic T_s/T_n are
    # robot-on-cart forces.
    force_on_cart_w = -state.hand_force_w
    projected_t_s = torch.sum(force_on_cart_w * base_env.path_tangent_w, dim=-1)
    projected_t_n = torch.sum(force_on_cart_w * base_env.path_normal_w, dim=-1)

    def relative_error(reference: Any, measured: Any) -> Any:
        denominator = torch.maximum(torch.maximum(torch.abs(reference), torch.abs(measured)), torch.ones_like(reference))
        return torch.abs(reference - measured) / denominator

    sign_active_s = (torch.abs(analytic.t_s) > 1.0) | (torch.abs(projected_t_s) > 1.0)
    sign_active_n = (torch.abs(analytic.t_n) > 1.0) | (torch.abs(projected_t_n) > 1.0)
    sign_s = torch.where(sign_active_s, torch.sign(analytic.t_s) == torch.sign(projected_t_s), torch.ones_like(sign_active_s))
    sign_n = torch.where(sign_active_n, torch.sign(analytic.t_n) == torch.sign(projected_t_n), torch.ones_like(sign_active_n))

    result = {
        "speed_error": speed_error,
        "overspeed": overspeed,
        "lateral_error": base_env.path_state.lateral_error,
        "heading_error": torch.atan2(torch.sin(base_env.path_state.heading_error), torch.cos(base_env.path_state.heading_error)),
        "pitch_error": pitch_error,
        "hitch_height_error": hitch_error,
        "two_wheel_contact": state.two_wheel_contact,
        "wheel_normal_force_left": state.wheel_normal_force[:, 0],
        "wheel_normal_force_right": state.wheel_normal_force[:, 1],
        "foot_slip": foot_slip,
        "processed_action_rate": action_rate,
        "processed_action_jerk": action_jerk,
        "power": power,
        "connection_residual": state.connection_residual,
        "connection_force": connection_force,
        "connection_torque": connection_torque,
        "connection_force_asymmetry": force_asymmetry,
        "connection_torque_asymmetry": torque_asymmetry,
        "t_s_relative_error": relative_error(analytic.t_s, projected_t_s),
        "t_n_relative_error": relative_error(analytic.t_n, projected_t_n),
        "t_s_sign_agreement": sign_s,
        "t_n_sign_agreement": sign_n,
        "analytic_force_valid": analytic.valid,
        "fat_wrench_consistent": stability.fat_wrench_consistent,
        "fat_wrench_t_s_relative_error": stability.fat_wrench_relative_error[:, 0],
        "fat_wrench_t_n_relative_error": stability.fat_wrench_relative_error[:, 1],
        "zmp_margin": stability.zmp_margin,
        "zmp_valid": stability.zmp_valid,
        "arm_torque_margin": arm_margin,
        "leg_torque_margin": leg_margin,
    }
    if teacher_kl is not None:
        result["teacher_student_kl"] = teacher_kl
    return result


def _labels(base_env: Any, env_ids: Any) -> tuple[list[str], list[str], list[str]]:
    stages = ["TRAINING"] * int(env_ids.numel())
    phases = command_phase_labels(
        base_env.command_manager.get_command("twist")[env_ids, 0].detach().cpu().numpy(),
    )
    return (
        stages,
        ["RANDOM"] * len(stages),
        phases,
    )


def _run_mode(
    env: Any,
    base_env: Any,
    policy: PolicyHandle,
    teacher: PolicyHandle | None,
    *,
    seeds: list[int],
    episodes: int,
    max_steps_per_seed: int,
) -> tuple[PolicyEvaluationAccumulator, dict[str, Any]]:
    import torch

    if not seeds:
        raise ValueError("fixed evaluation seeds must be non-empty")
    quota_divisor = len(seeds) * len(CROSS_CASE_LABELS)
    if episodes % quota_divisor != 0:
        raise ValueError("episodes must be divisible by the number of seeds")
    accumulator = PolicyEvaluationAccumulator()
    completed = 0
    in_flight = 0
    enrolled = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    episode_returns: list[float] = []
    episode_return = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_phases: list[set[str]] = [set() for _ in range(base_env.num_envs)]
    cause_names = tuple(base_env.termination_manager.active_terms)

    def reserve_episodes(candidate_ids: Any, needed: int) -> None:
        nonlocal in_flight

        if candidate_ids.numel() == 0:
            return
        selected = candidate_ids[~enrolled[candidate_ids]][:needed]
        enrolled[selected] = True
        in_flight += int(selected.numel())

    for seed_index, seed in enumerate(seeds):
        milestone = (seed_index + 1) * episodes // len(seeds)
        env.seed(seed)
        observation, _ = env.reset()
        observation = observation.to(base_env.device)
        if in_flight:
            raise RuntimeError("evaluation seed changed with unfinished reserved episodes")
        enrolled.zero_()

        reserve_episodes(
            torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long),
            milestone - completed - in_flight,
        )
        episode_return.zero_()
        for phases in episode_phases:
            phases.clear()
        policy_steps = 0
        while completed < milestone:
            if policy_steps >= max_steps_per_seed:
                raise RuntimeError(
                    f"seed {seed} exceeded {max_steps_per_seed} policy steps; "
                    f"remaining episodes={milestone - completed}; in_flight={in_flight}"
                )
            active = enrolled
            _apply_evaluation_command_protocol(base_env, active)
            with torch.no_grad():
                distribution = policy.distribution(observation)
                teacher_kl = None
                if teacher is not None:
                    teacher_distribution = teacher.distribution(observation)
                    teacher_kl = torch.distributions.kl_divergence(teacher_distribution, distribution)
                if torch.any(active):
                    raw_samples = _sample_metrics(base_env, teacher_kl)
                    ids = torch.nonzero(active, as_tuple=False).flatten()
                    samples = {
                        name: value[ids].detach().cpu().numpy()
                        for name, value in raw_samples.items()
                    }
                    stage_labels, case_labels, phase_labels = _labels(base_env, ids)
                    for env_id, phase in zip(
                        ids.detach().cpu().tolist(),
                        phase_labels,
                        strict=True,
                    ):
                        episode_phases[int(env_id)].add(str(phase))
                    accumulator.add_step(
                        samples,
                        stage_labels=stage_labels,
                        cross_case_labels=case_labels,
                        phase_labels=phase_labels,
                    )
                actions = distribution.mean
                observation, reward, dones, extras = env.step(actions)
                observation = observation.to(base_env.device)
            episode_return += reward * active.to(dtype=reward.dtype)
            done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
            if done_ids.numel() > 0:
                time_outs = extras["time_outs"]
                if not torch.is_tensor(time_outs) or time_outs.shape != dones.shape:
                    raise RuntimeError("evaluation step did not expose per-environment timeout flags")
                reusable_ids: list[int] = []
                for env_id in done_ids.detach().cpu().tolist():
                    if not bool(enrolled[env_id].item()) or not bool(active[env_id].item()):
                        episode_return[env_id] = 0.0
                        episode_phases[env_id].clear()
                        continue
                    value = float(episode_return[env_id].item())
                    causes = [
                        name
                        for name in cause_names
                        if bool(base_env.termination_manager.get_term(name)[env_id].item())
                    ]
                    fell = _episode_fell(
                        timed_out=bool(time_outs[env_id].item()),
                        causes=causes,
                    )
                    episode_returns.append(value)
                    completed += 1
                    in_flight -= 1
                    enrolled[env_id] = False
                    reusable_ids.append(env_id)
                    if not episode_phases[env_id]:
                        raise RuntimeError("active completed episode has no phase evidence")
                    accumulator.add_episode(
                        value,
                        fell=fell,
                        causes=causes,
                        phase_labels=[
                            label
                            for label in COMMAND_PHASE_LABELS
                            if label in episode_phases[env_id]
                        ],
                        cross_case_label="RANDOM",
                    )
                    episode_return[env_id] = 0.0
                    episode_phases[env_id].clear()
                if reusable_ids:
                    reserve_episodes(
                        torch.tensor(
                            reusable_ids, device=base_env.device, dtype=torch.long
                        ),
                        milestone - completed - in_flight,
                    )
            policy_steps += 1

        if (
            in_flight
            or torch.any(enrolled)
        ):
            raise RuntimeError("evaluation milestone completed with reserved episodes still active")

    if completed != episodes:
        raise RuntimeError(f"episode quota drifted: {completed}")
    return_summary = {
        "episodes": len(episode_returns),
        "mean": sum(episode_returns) / len(episode_returns),
    }
    return accumulator, return_summary


def _checkpoint_binding(path: Path, checkpoint: Any) -> dict[str, Any]:
    return {
        "path": os.fspath(path),
        "stage": checkpoint[CHECKPOINT_STAGE_KEY],
        "curriculum_iteration": checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY),
    }


def main() -> int:  # noqa: C901
    parser = _parser()
    args = parser.parse_args()
    checkpoint_path = require_existing_file(args.checkpoint, "policy checkpoint").resolve()
    checkpoint = load_stage_checkpoint(
        checkpoint_path,
        expected_stage=SUPPORTED_STAGES,
        validate_runtime=False,
    )
    stage = checkpoint[CHECKPOINT_STAGE_KEY]
    training_configuration = dict(checkpoint[TRAINING_CONFIGURATION_KEY])
    if training_configuration["task"] != args.task:
        raise ValueError("policy evaluation task differs from checkpoint training task")
    training_parameters = training_configuration["training_parameters"]
    rollout_steps = int(training_parameters["rollout_steps"])
    latent_dim = int(training_parameters["latent_dim"])
    history_length = int(training_parameters["history_length"])
    _validate_args(args, stage)
    is_student = stage != "s0_teacher"

    s1_baseline_path: Path | None = None
    s1_baseline_returns: dict[str, float] | None = None
    s1_baseline_binding: dict[str, Any] | None = None
    if args.s1_baseline_report is not None:
        s1_baseline_path = require_existing_file(
            args.s1_baseline_report, "S1 baseline diagnostic report"
        ).resolve()
        s1_report = json.loads(s1_baseline_path.read_text(encoding="utf-8"))
        s1_baseline_returns = validate_s1_baseline_diagnostic_report(
            s1_report,
            fixed_seeds=args.seeds,
            episodes_per_stage=args.episodes,
        )
        baseline_parameters = s1_report["evaluation"]
        if (
            baseline_parameters.get("latent_dim") != latent_dim
            or baseline_parameters.get("history_length") != history_length
            or baseline_parameters.get("rollout_steps") != rollout_steps
        ):
            raise ValueError("S1 baseline uses different training parameters")
        s1_baseline_binding = {
            "path": os.fspath(s1_baseline_path),
            "baseline_return_mean": dict(s1_baseline_returns),
        }

    teacher_path: Path | None = None
    teacher_checkpoint: Any | None = None
    if args.teacher_checkpoint is not None:
        teacher_path = require_existing_file(args.teacher_checkpoint, "teacher checkpoint").resolve()
        teacher_checkpoint = load_stage_checkpoint(
            teacher_path, expected_stage="s0_teacher", validate_runtime=False
        )
        if teacher_checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"] != training_parameters:
            raise ValueError("teacher checkpoint uses different training parameters")

    stage_reports: dict[str, Any] = {}
    omitted_diagnostics: list[str] = []
    try:
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper

        import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity  # noqa: F401

        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        raw_env = None
        env = None
        try:
            env_cfg, _ = load_mjlab_configs(
                args.task,
                play=True,
                num_envs=args.num_envs,
                seed=args.seeds[0],
                history_length=history_length,
            )
            _configure_evaluation(env_cfg)
            if is_student and teacher_path is None:
                env_cfg.observations.pop("teacher_dynamic_history", None)
                env_cfg.observations.pop("teacher_static", None)
            agent_key = "rsl_rl_cfg_entry_point" if stage == "s0_teacher" else "rsl_rl_student_cfg_entry_point"
            agent_cfg = _load_rsl_runner_cfg(
                args.task,
                agent_key,
                device,
                latent_dim,
                history_length,
            )
            raw_env = ManagerBasedRlEnv(env_cfg, device=device)
            env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
            base_env = raw_env.unwrapped
            policy, keepalive = _load_policy(
                env, checkpoint_path, checkpoint, stage, device, args.task
            )
            teacher: PolicyHandle | None = None
            if is_student and teacher_path is not None:
                teacher, teacher_keepalive = _load_teacher_policy(
                    env, teacher_path, device, args.task
                )
                keepalive.extend(teacher_keepalive)
            del keepalive  # Actors/runners remain reachable through PolicyHandle objects.

            baseline_accumulator, baseline_return = _run_mode(
                env,
                base_env,
                policy,
                teacher,
                seeds=list(args.seeds),
                episodes=args.episodes,
                max_steps_per_seed=args.max_policy_steps_per_seed,
            )
            metrics = baseline_accumulator.summary()
            stratified = baseline_accumulator.stratified_summary()
            stage_reports[EVALUATION_STAGE] = {
                "metrics": metrics,
                "stratified": stratified,
                "return": baseline_return,
            }
        finally:
            if env is not None:
                env.close()
            elif raw_env is not None:
                raw_env.close()
    except BaseException:
        raise

    if is_student and teacher_checkpoint is None:
        omitted_diagnostics.append("teacher_student_action_kl")
    stage_report = stage_reports[EVALUATION_STAGE]
    if stage_report["metrics"]["episodes"]["completed"] != args.episodes:
        raise RuntimeError("training evaluation episode quota is incomplete")

    if stage == "s2_student_ppo":
        if s1_baseline_returns is None:
            omitted_diagnostics.append("s1_baseline_return_comparison")
        else:
            comparisons = evaluate_s2_return_floor(stage_reports, s1_baseline_returns)
            assert s1_baseline_binding is not None
            s1_baseline_binding["s2_return_floor"] = comparisons

    report: dict[str, Any] = {
        "schema_version": POLICY_DIAGNOSTIC_SCHEMA_VERSION,
        "report_type": "g1_rickshaw_policy_diagnostics",
        "status": "recorded",
        "created_utc": utc_timestamp(),
        "task": args.task,
        "checkpoint": _checkpoint_binding(checkpoint_path, checkpoint),
        "teacher_checkpoint": (
            None
            if teacher_path is None or teacher_checkpoint is None
            else _checkpoint_binding(teacher_path, teacher_checkpoint)
        ),
        "s1_baseline": s1_baseline_binding,
        "evaluation": {
            "deterministic_actions": True,
            "fixed_seeds": list(args.seeds),
            "episodes_per_stage": args.episodes,
            "num_envs": args.num_envs,
            "curriculum_stages": [EVALUATION_STAGE],
            "command_protocol": FORMAL_EVALUATION_COMMAND_PROTOCOL,
            "cross_case_protocol": FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
            "rollout_steps": rollout_steps,
            "latent_dim": latent_dim,
            "history_length": history_length,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "stages": stage_reports,
        "omitted_diagnostics": omitted_diagnostics,
    }
    write_json_atomic(args.output, report)

    print(f"wrote policy diagnostic report: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
