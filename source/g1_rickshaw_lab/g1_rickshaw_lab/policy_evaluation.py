"""Pure policy-diagnostic aggregation and artifact contracts.

The simulator-facing runner lives in ``scripts/evaluate_policy.py``.  Keeping
the reductions here free of simulator imports makes every reported diagnostic
number independently testable on CPU.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

POLICY_DIAGNOSTIC_SCHEMA_VERSION: Final[int] = 6
GUIDE_POLICY_EVALUATION_TASK: Final[str] = "Mjlab-G1-Rickshaw-Slopes-Student"
COMMAND_PHASE_LABELS: Final[tuple[str, ...]] = (
    "standing",
    "moving",
)
FORMAL_EVALUATION_COMMAND_PROTOCOL: Final[str] = "deterministic_0_to_1_to_0_mps"
METRIC_DEFINITIONS: Final[dict[str, str]] = {
    "tracking.lin_vel_x_rmse_mps": "rickshaw forward-speed RMSE over policy samples",
    "tracking.ang_vel_z_rmse_radps": "rickshaw yaw-rate RMSE over policy samples",
    "episodes.fall_rate": "non-timeout terminated episodes / completed episodes",
    "tracking.overspeed_rate": "samples with v_s > v_ref + configured safety margin / samples",
    "rickshaw.pitch_error": "actual path-frame pitch minus alpha_target",
    "rickshaw.hitch_height_error": "actual world-vertical hitch height minus target",
    "rickshaw.two_wheel_contact_rate": "samples where both wheel normal forces pass the safety threshold",
    "rickshaw.wheel_normal_force": "per-wheel force percentiles",
    "locomotion.foot_slip": "summed ground-plane speed of contacting feet",
    "actions.normalized_rate": "RMS normalized-action rate over policy joints",
    "actions.normalized_jerk": "RMS normalized-action second derivative over policy joints",
    "actuation.power": "sum(abs(actuator_force * joint_velocity)) over the 29 policy joints",
    "connection.residual": "maximum position residual of the two MuJoCo site connections",
    "connection.force": "resultant robot-on-cart hand-force norm",
    "actuation.arm/leg_torque_margin": ("minimum per-environment 1-|actuator_force|/current actuator.effort_limit"),
    "distillation.teacher_student_action_kl": "KL(teacher Gaussian || student Gaussian), summed over 29 actions",
    "stratified": ("full metric reductions by command phase under the deterministic 0->1->0 m/s protocol"),
}


def command_phase_labels(
    v_ref: Any,
    *,
    velocity_epsilon: float = 1.0e-3,
) -> list[str]:
    """Classify direct Mjlab-style speed commands as standing or moving."""

    if velocity_epsilon < 0.0:
        raise ValueError("command phase epsilon must be non-negative")
    velocity = np.asarray(v_ref, dtype=np.float64)
    if velocity.ndim != 1 or not np.all(np.isfinite(velocity)):
        raise ValueError("v_ref must be a finite one-dimensional array")
    labels = np.full(velocity.shape, "moving", dtype=object)
    labels[np.abs(velocity) <= velocity_epsilon] = "standing"
    return labels.tolist()


def validate_stratified_summary(value: Any, *, label: str = "stratified") -> None:
    """Require the complete command-phase evaluation grid in an artifact."""

    if not isinstance(value, Mapping) or set(value) != {"by_phase"}:
        raise ValueError(f"{label} must contain the exact stratified reductions")

    def validate_leaves(raw: Any, expected: Sequence[str], *, leaf_label: str) -> None:
        if not isinstance(raw, Mapping) or set(raw) != set(expected):
            raise ValueError(f"{leaf_label} has an incomplete label set")
        for name, summary in raw.items():
            samples = summary.get("samples") if isinstance(summary, Mapping) else None
            if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
                raise ValueError(f"{leaf_label}.{name} has an invalid sample count")
            episodes = summary.get("episodes")
            completed = episodes.get("completed") if isinstance(episodes, Mapping) else None
            fall_rate = episodes.get("fall_rate") if isinstance(episodes, Mapping) else None
            causes = episodes.get("termination_cause_histogram") if isinstance(episodes, Mapping) else None
            if (
                isinstance(completed, bool)
                or not isinstance(completed, int)
                or completed <= 0
                or isinstance(fall_rate, bool)
                or not isinstance(fall_rate, (int, float))
                or not math.isfinite(float(fall_rate))
                or not isinstance(causes, Mapping)
            ):
                raise ValueError(f"{leaf_label}.{name} has incomplete episode evidence")

    validate_leaves(value["by_phase"], COMMAND_PHASE_LABELS, leaf_label=f"{label}.by_phase")


def validate_s1_baseline_diagnostic_report(
    report: Any,
    *,
    fixed_seeds: Sequence[int],
    episodes_per_stage: int,
) -> dict[str, float]:
    """Validate an S1 return baseline used for an S2 diagnostic comparison."""

    if not isinstance(report, Mapping):
        raise ValueError("S1 baseline diagnostic report must be a mapping")
    if report.get("schema_version") != POLICY_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("S1 baseline diagnostic report schema is unsupported")
    if report.get("report_type") != "g1_rickshaw_policy_diagnostics":
        raise ValueError("S1 baseline report has the wrong report_type")
    if report.get("task") != GUIDE_POLICY_EVALUATION_TASK:
        raise ValueError("S1 baseline report does not use the Guide training task")
    if report.get("status") != "recorded":
        raise ValueError("S1 baseline diagnostic report is incomplete")
    checkpoint = report.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("stage") != "s1_context_distillation"
        or not isinstance(checkpoint.get("path"), str)
    ):
        raise ValueError("S1 baseline report checkpoint binding differs from S2 lineage")
    evaluation = report.get("evaluation")
    expected_seeds = list(fixed_seeds)
    if (
        isinstance(episodes_per_stage, bool)
        or not isinstance(episodes_per_stage, int)
        or episodes_per_stage <= 0
        or not expected_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in expected_seeds)
        or len(set(expected_seeds)) != len(expected_seeds)
        or episodes_per_stage % len(expected_seeds) != 0
    ):
        raise ValueError("S1/S2 diagnostic episode quota must be positive and divisible by the number of fixed seeds")
    curriculum_stages = evaluation.get("curriculum_stages") if isinstance(evaluation, Mapping) else None
    num_envs = evaluation.get("num_envs") if isinstance(evaluation, Mapping) else None
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("deterministic_actions") is not True
        or evaluation.get("fixed_seeds") != expected_seeds
        or evaluation.get("episodes_per_stage") != episodes_per_stage
        or isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or evaluation.get("command_protocol") != FORMAL_EVALUATION_COMMAND_PROTOCOL
        or not isinstance(curriculum_stages, (list, tuple))
        or list(curriculum_stages) != ["training"]
    ):
        raise ValueError("S1 baseline report does not use the exact S2 seeds and episode quota")

    stages = report.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("S1 baseline report is missing curriculum-stage results")
    returns: dict[str, float] = {}
    for stage_name in ("training",):
        stage_report = stages.get(stage_name)
        if not isinstance(stage_report, Mapping):
            raise ValueError(f"S1 baseline report is missing {stage_name} results")
        validate_stratified_summary(stage_report.get("stratified"), label=f"S1 {stage_name}.stratified")
        baseline = stage_report.get("return")
        if not isinstance(baseline, Mapping):
            raise ValueError(f"S1 {stage_name} report has no baseline return")
        baseline_episodes = baseline.get("episodes")
        if (
            isinstance(baseline_episodes, bool)
            or not isinstance(baseline_episodes, int)
            or baseline_episodes < episodes_per_stage
        ):
            raise ValueError(f"S1 {stage_name} baseline return episode quota is incomplete")
        mean = baseline.get("mean")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean):
            raise ValueError(f"S1 {stage_name} baseline mean return is not finite")
        returns[stage_name] = float(mean)
    return returns


def evaluate_s2_return_floor(
    stage_reports: Mapping[str, Any],
    s1_baseline_returns: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Compare fixed-seed S2 TRAINING return with the S1 baseline."""

    comparisons: dict[str, dict[str, Any]] = {}
    for stage_name in ("training",):
        stage_report = stage_reports.get(stage_name)
        baseline = stage_report.get("return") if isinstance(stage_report, Mapping) else None
        s2_return = baseline.get("mean") if isinstance(baseline, Mapping) else None
        s1_return = s1_baseline_returns.get(stage_name)
        if (
            isinstance(s1_return, bool)
            or not isinstance(s1_return, (int, float))
            or not math.isfinite(s1_return)
            or isinstance(s2_return, bool)
            or not isinstance(s2_return, (int, float))
            or not math.isfinite(s2_return)
        ):
            raise ValueError(f"S1/S2 {stage_name} baseline return comparison is incomplete")
        delta = float(s2_return) - float(s1_return)
        comparisons[stage_name] = {
            "s1_baseline_mean": float(s1_return),
            "s2_baseline_mean": float(s2_return),
            "delta": delta,
            "meets_or_exceeds_s1": delta >= 0.0,
        }
    return comparisons


def _as_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"sample {name!r} must be one-dimensional, got {result.shape}")
    if result.dtype == np.bool_:
        return result.astype(np.float32)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"sample {name!r} must be numeric")
    return result.astype(np.float32, copy=False)


def _rms(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values, dtype=np.float64))


def _maximum_absolute(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.max(np.abs(values)))


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("p01", "p05", "p50", "p90", "p95", "p99", "max")}
    quantiles = np.quantile(values, (0.01, 0.05, 0.50, 0.90, 0.95, 0.99))
    return {
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(np.max(values)),
    }


@dataclass
class MetricStore:
    """Chunked sample/episode store used by evaluation reductions."""

    chunks: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    non_finite_counts: Counter[str] = field(default_factory=Counter)
    episode_returns: list[float] = field(default_factory=list)
    falls: int = 0
    termination_causes: Counter[str] = field(default_factory=Counter)

    def add_samples(self, samples: Mapping[str, Any]) -> None:
        """Append an equally-sized batch, excluding but accounting for NaN/Inf."""

        expected: int | None = None
        for name, raw_value in samples.items():
            values = _as_vector(raw_value, name)
            if expected is None:
                expected = values.size
            elif values.size != expected:
                raise ValueError("all sample arrays in one batch must have equal length")
            finite = np.isfinite(values)
            self.non_finite_counts[name] += int(np.sum(~finite))
            if np.any(finite):
                self.chunks[name].append(values[finite].copy())

    def add_episode(self, episode_return: float, *, fell: bool, causes: Sequence[str]) -> None:
        value = float(episode_return)
        if not math.isfinite(value):
            raise ValueError("episode return must be finite")
        self.episode_returns.append(value)
        self.falls += int(fell)
        self.termination_causes.update(str(cause) for cause in causes)

    def values(self, name: str) -> np.ndarray:
        chunks = self.chunks.get(name, ())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def summary(self) -> dict[str, Any]:  # noqa: C901 - mirrors the guide's metric list.
        lin_vel_x_error = self.values("lin_vel_x_error")
        ang_vel_z_error = self.values("ang_vel_z_error")
        pitch = self.values("pitch_error")
        hitch = self.values("hitch_height_error")
        returns = np.asarray(self.episode_returns, dtype=np.float32)
        episodes = len(self.episode_returns)

        def distribution(name: str) -> dict[str, Any]:
            values = self.values(name)
            return {"mean": _mean(values), **_percentiles(values)}

        summary = {
            "samples": int(lin_vel_x_error.size),
            "non_finite_sample_counts": dict(sorted(self.non_finite_counts.items())),
            "episodes": {
                "completed": episodes,
                "falls": self.falls,
                "fall_rate": None if episodes == 0 else self.falls / episodes,
                "return": {"mean": _mean(returns), **_percentiles(returns)},
                "termination_cause_histogram": dict(sorted(self.termination_causes.items())),
            },
            "tracking": {
                "lin_vel_x_rmse_mps": _rms(lin_vel_x_error),
                "ang_vel_z_rmse_radps": _rms(ang_vel_z_error),
                "overspeed_rate": _mean(self.values("overspeed")),
            },
            "rickshaw": {
                "pitch_error": {"rms_rad": _rms(pitch), "max_abs_rad": _maximum_absolute(pitch)},
                "hitch_height_error": {"rms_m": _rms(hitch), "max_abs_m": _maximum_absolute(hitch)},
                "two_wheel_contact_rate": _mean(self.values("two_wheel_contact")),
                "wheel_normal_force_n": {
                    "left": distribution("wheel_normal_force_left"),
                    "right": distribution("wheel_normal_force_right"),
                },
            },
            "locomotion": {"foot_slip_mps": distribution("foot_slip")},
            "actions": {
                "normalized_rate_per_s": distribution("normalized_action_rate"),
                "normalized_jerk_per_s2": distribution("normalized_action_jerk"),
            },
            "actuation": {
                "power_w": distribution("power"),
                "arm_torque_margin": distribution("arm_torque_margin"),
                "leg_torque_margin": distribution("leg_torque_margin"),
            },
            "connection": {
                "residual_m": distribution("connection_residual"),
                "force_n": distribution("connection_force"),
            },
            "distillation": {
                "teacher_student_action_kl": distribution("teacher_student_kl"),
            },
        }
        return summary


@dataclass
class PolicyEvaluationAccumulator:
    """Global metrics with command-phase reductions."""

    global_store: MetricStore = field(default_factory=MetricStore)
    phase_stores: dict[str, MetricStore] = field(default_factory=dict)

    @staticmethod
    def _add_labeled_samples(
        stores: dict[str, MetricStore],
        labels: Sequence[str],
        vectors: Mapping[str, np.ndarray],
    ) -> None:
        label_array = np.asarray(labels, dtype=object)
        for label in dict.fromkeys(str(value) for value in labels):
            selected = label_array == label
            stores.setdefault(label, MetricStore()).add_samples(
                {name: value[selected] for name, value in vectors.items()}
            )

    def add_step(
        self,
        samples: Mapping[str, Any],
        *,
        phase_labels: Sequence[str] | None = None,
    ) -> None:
        vectors = {name: _as_vector(value, name) for name, value in samples.items()}
        sizes = {value.size for value in vectors.values()}
        if len(sizes) != 1:
            raise ValueError("sample arrays must have equal length")
        batch_size = sizes.pop() if sizes else 0
        self.global_store.add_samples(vectors)
        if phase_labels is not None:
            if len(phase_labels) != batch_size:
                raise ValueError("command-phase labels have the wrong length")
            unknown = sorted(set(phase_labels) - set(COMMAND_PHASE_LABELS))
            if unknown:
                raise ValueError(f"unknown command-phase labels: {unknown}")
            self._add_labeled_samples(self.phase_stores, phase_labels, vectors)

    def add_episode(
        self,
        episode_return: float,
        *,
        fell: bool,
        causes: Sequence[str],
        phase_labels: Sequence[str],
    ) -> None:
        observed_phases = tuple(dict.fromkeys(str(label) for label in phase_labels))
        if not observed_phases:
            raise ValueError("episode must contain at least one command phase")
        unknown_phases = sorted(set(observed_phases) - set(COMMAND_PHASE_LABELS))
        if unknown_phases:
            raise ValueError(f"unknown command-phase labels {unknown_phases}")
        self.global_store.add_episode(episode_return, fell=fell, causes=causes)
        for phase_label in observed_phases:
            self.phase_stores.setdefault(phase_label, MetricStore()).add_episode(
                episode_return, fell=fell, causes=causes
            )

    def summary(self) -> dict[str, Any]:
        return self.global_store.summary()

    def stratified_summary(self) -> dict[str, Any]:
        def summarize(stores: Mapping[str, MetricStore], labels: Sequence[str]) -> dict[str, Any]:
            return {label: stores.get(label, MetricStore()).summary() for label in labels}

        return {
            "by_phase": summarize(self.phase_stores, COMMAND_PHASE_LABELS),
        }


__all__ = [
    "COMMAND_PHASE_LABELS",
    "FORMAL_EVALUATION_COMMAND_PROTOCOL",
    "GUIDE_POLICY_EVALUATION_TASK",
    "METRIC_DEFINITIONS",
    "POLICY_DIAGNOSTIC_SCHEMA_VERSION",
    "MetricStore",
    "PolicyEvaluationAccumulator",
    "command_phase_labels",
    "evaluate_s2_return_floor",
    "validate_s1_baseline_diagnostic_report",
    "validate_stratified_summary",
]
