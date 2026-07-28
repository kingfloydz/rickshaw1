"""Feasibility configuration validation used by the Mjlab runtime."""

from __future__ import annotations

import pytest

from g1_rickshaw_lab.configuration import (
    FEASIBILITY_SCHEMA_VERSION,
    G1_JOINT_ORDER,
    REQUIRED_CALIBRATION_FIELDS,
    REQUIRED_FEASIBILITY_RANGES,
    FeasibilityConfigError,
    FeasibilityEnvelope,
)


def _range_for(name: str) -> dict[str, float]:
    if name.startswith("payload.com."):
        return {"min": -0.05, "max": 0.05}
    if name == "payload.mass":
        return {"min": 0.0, "max": 10.0}
    if name == "terrain.friction":
        return {"min": 0.6, "max": 1.2}
    return {"min": 0.01, "max": 10.0}


def _valid_mapping() -> dict:
    return {
        "schema_version": FEASIBILITY_SCHEMA_VERSION,
        "joint_order": list(G1_JOINT_ORDER),
        "ranges": {name: _range_for(name) for name in REQUIRED_FEASIBILITY_RANGES},
        "calibration": {name: 1.0 for name in REQUIRED_CALIBRATION_FIELDS},
    }


def test_feasibility_envelope_requires_exact_schema_fields() -> None:
    envelope = FeasibilityEnvelope.from_mapping(_valid_mapping())
    assert tuple(envelope.joint_order) == G1_JOINT_ORDER
    assert set(envelope.ranges) == set(REQUIRED_FEASIBILITY_RANGES)
    assert set(envelope.calibration) == set(REQUIRED_CALIBRATION_FIELDS)

    missing = _valid_mapping()
    del missing["ranges"]["torso.mass_delta"]
    with pytest.raises(FeasibilityConfigError, match="missing"):
        FeasibilityEnvelope.from_mapping(missing)

    unknown = _valid_mapping()
    unknown["calibration"]["unvalidated.default"] = 1.0
    with pytest.raises(FeasibilityConfigError, match="unknown"):
        FeasibilityEnvelope.from_mapping(unknown)


def test_feasibility_envelope_rejects_joint_order_drift() -> None:
    mapping = _valid_mapping()
    mapping["joint_order"][0], mapping["joint_order"][1] = (
        mapping["joint_order"][1],
        mapping["joint_order"][0],
    )
    with pytest.raises(FeasibilityConfigError, match="fixed policy joint order"):
        FeasibilityEnvelope.from_mapping(mapping)
