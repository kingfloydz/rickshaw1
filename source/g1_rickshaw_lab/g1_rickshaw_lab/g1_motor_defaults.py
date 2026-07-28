"""Unitree G1 motor parameters resolved from MJLab v1.5.3."""

from __future__ import annotations

import re
from types import MappingProxyType

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    DAMPING_RATIO,
    G1_ARTICULATION,
    NATURAL_FREQ,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
)

from .configuration import G1_JOINT_ORDER

NATURAL_FREQUENCY = NATURAL_FREQ


def _parameters_for_joint(name: str) -> tuple[float, float, float, float]:
    matches = [
        actuator
        for actuator in G1_ARTICULATION.actuators
        if any(re.fullmatch(pattern, name) for pattern in actuator.target_names_expr)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"MJLab G1 articulation maps {name!r} to {len(matches)} actuators")
    actuator = matches[0]
    return (
        float(actuator.stiffness),
        float(actuator.damping),
        float(actuator.effort_limit),
        float(actuator.armature),
    )


G1_MOTOR_PARAMETERS_BY_JOINT = MappingProxyType({name: _parameters_for_joint(name) for name in G1_JOINT_ORDER})
G1_JOINT_STIFFNESS = tuple(value[0] for value in G1_MOTOR_PARAMETERS_BY_JOINT.values())
G1_JOINT_DAMPING = tuple(value[1] for value in G1_MOTOR_PARAMETERS_BY_JOINT.values())
G1_JOINT_EFFORT_LIMITS = tuple(value[2] for value in G1_MOTOR_PARAMETERS_BY_JOINT.values())
G1_JOINT_ARMATURE = tuple(value[3] for value in G1_MOTOR_PARAMETERS_BY_JOINT.values())
G1_ACTION_SCALE = tuple(
    0.25 * effort_limit / stiffness for stiffness, _, effort_limit, _ in G1_MOTOR_PARAMETERS_BY_JOINT.values()
)


__all__ = [
    "ARMATURE_4010",
    "ARMATURE_5020",
    "ARMATURE_7520_14",
    "ARMATURE_7520_22",
    "DAMPING_4010",
    "DAMPING_5020",
    "DAMPING_7520_14",
    "DAMPING_7520_22",
    "DAMPING_RATIO",
    "G1_ACTION_SCALE",
    "G1_ARTICULATION",
    "G1_JOINT_ARMATURE",
    "G1_JOINT_DAMPING",
    "G1_JOINT_EFFORT_LIMITS",
    "G1_JOINT_STIFFNESS",
    "G1_MOTOR_PARAMETERS_BY_JOINT",
    "NATURAL_FREQUENCY",
    "STIFFNESS_4010",
    "STIFFNESS_5020",
    "STIFFNESS_7520_14",
    "STIFFNESS_7520_22",
]
