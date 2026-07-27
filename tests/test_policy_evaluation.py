"""CPU-only tests for policy diagnostic aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from g1_rickshaw_lab.policy_evaluation import (
    MetricStore,
    PolicyEvaluationAccumulator,
    command_phase_labels,
    evaluate_s2_return_floor,
)


def test_s2_return_comparison_is_diagnostic_only() -> None:
    comparisons = evaluate_s2_return_floor(
        {"training": {"return": {"mean": 1.25}}},
        {"training": 1.5},
    )
    assert comparisons == {
        "training": {
            "s1_baseline_mean": 1.5,
            "s2_baseline_mean": 1.25,
            "delta": -0.25,
            "meets_or_exceeds_s1": False,
        }
    }


def test_command_phase_labels_are_deterministic() -> None:
    assert command_phase_labels([0.0, 0.5, 1.0]) == [
        "standing",
        "moving",
        "moving",
    ]


def test_metric_store_excludes_nonfinite_samples_but_records_them() -> None:
    store = MetricStore()
    store.add_samples(
        {
            "lin_vel_x_error": np.asarray([1.0, np.nan, -1.0]),
            "ang_vel_z_error": np.asarray([0.2, np.nan, -0.2]),
            "overspeed": np.asarray([0.0, 1.0, 0.0]),
        }
    )
    store.add_episode(2.0, fell=False, causes=("timeout",))
    summary = store.summary()
    assert summary["samples"] == 2
    assert summary["non_finite_sample_counts"] == {
        "overspeed": 0,
        "ang_vel_z_error": 1,
        "lin_vel_x_error": 1,
    }
    assert summary["tracking"]["lin_vel_x_rmse_mps"] == pytest.approx(1.0)
    assert summary["tracking"]["ang_vel_z_rmse_radps"] == pytest.approx(0.2)
    assert summary["episodes"]["return"]["mean"] == pytest.approx(2.0)


def test_accumulator_keeps_global_and_phase_diagnostics() -> None:
    accumulator = PolicyEvaluationAccumulator()
    accumulator.add_step(
        {
            "lin_vel_x_error": [0.1, -0.2],
            "ang_vel_z_error": [0.0, 0.1],
            "overspeed": [0.0, 1.0],
        },
        phase_labels=["standing", "moving"],
    )
    accumulator.add_episode(
        3.0,
        fell=False,
        causes=("timeout",),
        phase_labels=("moving",),
    )
    global_summary = accumulator.summary()
    assert global_summary["samples"] == 2
    assert accumulator.stratified_summary()["by_phase"]["moving"]["samples"] == 1


def test_accumulator_rejects_mismatched_sample_lengths() -> None:
    accumulator = PolicyEvaluationAccumulator()
    with pytest.raises(ValueError, match="equal length"):
        accumulator.add_step({"lin_vel_x_error": [0.0], "overspeed": [0.0, 1.0]})
