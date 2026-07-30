"""Validated feasibility configuration for the G1 rickshaw task.

The feasibility file is a generated artifact, not a place for fallback
values. Loading it therefore performs complete validation
before returning an object that can be used by training or export code.

Canonical ``feasibility_envelope.yaml`` layout::

    schema_version: 7
    joint_order: [29 exact G1 joint names]
    ranges:
      rickshaw.mass_delta: {min: -20.0, max: 40.0}
      # all names in REQUIRED_FEASIBILITY_RANGES are required
    calibration:
      rolling_resistance.c_rr_nominal: 0.02
      # all names in REQUIRED_CALIBRATION_FIELDS are required

Nested mappings are accepted in ``ranges`` and ``calibration`` and are
flattened with dots.  An interval may be written as ``{min: x, max: y}`` or as
``[x, y]``.  The canonical file uses the mapping form.

"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

FEASIBILITY_SCHEMA_VERSION = 7

# This is the source-URDF order after applying the guide's one-time grouping
# rule: lower_names + waist_names + arm_names.  Runtime regex ordering is never
# used as a fixed policy interface.
G1_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
# Marginal bounds produced by the feasibility scan for runtime domain parameters.
REQUIRED_FEASIBILITY_RANGES = (
    "torso.mass_delta",
    "rickshaw.mass_delta",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
)

# Values that the guide explicitly leaves MISSING until asset inspection,
# hardware specification, calibration, or feasibility validation has run.
# Vector lengths are validated below; scalar fields must be finite.
REQUIRED_CALIBRATION_FIELDS = (
    "rolling_resistance.c_rr_nominal",
    "terrain.friction_nominal",
    "safety.overspeed_margin",
)

_CALIBRATION_STRICTLY_POSITIVE = frozenset(
    {
        "rolling_resistance.c_rr_nominal",
        "terrain.friction_nominal",
        "safety.overspeed_margin",
    }
)

_NONNEGATIVE_RANGE_NAMES = frozenset(
    name
    for name in REQUIRED_FEASIBILITY_RANGES
    if name
    not in {
        "torso.mass_delta",
        "rickshaw.mass_delta",
    }
)

_NOMINAL_CALIBRATION_BY_RANGE = {
    "rolling_resistance.c_rr": "rolling_resistance.c_rr_nominal",
    "terrain.friction": "terrain.friction_nominal",
}


class FeasibilityConfigError(ValueError):
    """Raised when a generated configuration violates the feasibility schema."""


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeasibilityConfigError(f"{path} must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise FeasibilityConfigError(f"{path} must be finite, got {value!r}")
    return result


def _expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeasibilityConfigError(f"{path} must be a mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise FeasibilityConfigError(f"{path} keys must be non-empty strings")
    return value


def _expect_exact_keys(mapping: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unknown={extra}")
        raise FeasibilityConfigError(f"invalid {path} fields: " + ", ".join(details))


def validate_joint_order(joint_order: Iterable[str], *, path: str = "joint_order") -> tuple[str, ...]:
    if isinstance(joint_order, (str, bytes)):
        raise FeasibilityConfigError(f"{path} must be a sequence of joint names")
    try:
        names = tuple(joint_order)
    except TypeError as exc:
        raise FeasibilityConfigError(f"{path} must be iterable") from exc
    if not all(isinstance(name, str) and name for name in names):
        raise FeasibilityConfigError(f"{path} must contain non-empty strings")
    if len(names) != 29:
        raise FeasibilityConfigError(f"{path} must contain exactly 29 joints, got {len(names)}")
    if len(set(names)) != len(names):
        raise FeasibilityConfigError(f"{path} contains duplicate joint names")
    if names != G1_JOINT_ORDER:
        mismatch = next(
            (
                index,
                expected,
                actual,
            )
            for index, (expected, actual) in enumerate(zip(G1_JOINT_ORDER, names, strict=True))
            if expected != actual
        )
        index, expected, actual = mismatch
        raise FeasibilityConfigError(f"{path}[{index}] is {actual!r}; fixed policy joint order requires {expected!r}")
    return names


@dataclass(frozen=True, slots=True)
class NumericRange:
    """A finite closed interval used by feasibility and training configs."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = _finite_float(self.minimum, "range.min")
        maximum = _finite_float(self.maximum, "range.max")
        if minimum > maximum:
            raise FeasibilityConfigError(f"range min {minimum} exceeds max {maximum}")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def from_value(cls, value: Any, *, path: str = "range") -> NumericRange:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            interval = _expect_mapping(value, path)
            _expect_exact_keys(interval, {"min", "max"}, path)
            return cls(interval["min"], interval["max"])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise FeasibilityConfigError(f"{path} interval sequence must have length 2")
            return cls(value[0], value[1])
        return cls(value, value)


def _looks_like_interval(value: Any) -> bool:
    if isinstance(value, NumericRange):
        return True
    if isinstance(value, Mapping):
        return set(value) == {"min", "max"}
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2


def _flatten_interval_mapping(value: Mapping[str, Any], *, path: str = "") -> dict[str, NumericRange]:
    result: dict[str, NumericRange] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key or key.startswith(".") or key.endswith("."):
            raise FeasibilityConfigError("range keys must be non-empty dotted identifiers")
        name = f"{path}.{key}" if path else key
        if _looks_like_interval(child):
            result[name] = NumericRange.from_value(child, path=f"ranges.{name}")
        elif not isinstance(child, Mapping):
            raise FeasibilityConfigError(f"ranges.{name} must be an explicit {{min, max}} or [min, max] interval")
        else:
            result.update(_flatten_interval_mapping(child, path=name))
    return result


def _flatten_values(value: Mapping[str, Any], *, path: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key or key.startswith(".") or key.endswith("."):
            raise FeasibilityConfigError("calibration keys must be non-empty dotted identifiers")
        name = f"{path}.{key}" if path else key
        if isinstance(child, Mapping):
            result.update(_flatten_values(child, path=name))
        else:
            result[name] = child
    return result


def _validate_calibration(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    flattened = _flatten_values(calibration)
    _expect_exact_keys(flattened, set(REQUIRED_CALIBRATION_FIELDS), "calibration")
    validated: dict[str, Any] = {}
    for name in REQUIRED_CALIBRATION_FIELDS:
        scalar = _finite_float(flattened[name], f"calibration.{name}")
        if name in _CALIBRATION_STRICTLY_POSITIVE and scalar <= 0.0:
            raise FeasibilityConfigError(f"calibration.{name} must be positive")
        validated[name] = scalar
    return MappingProxyType(validated)


@dataclass(frozen=True, slots=True)
class FeasibilityEnvelope:
    """Validated feasibility scan output and runtime range authority."""

    ranges: Mapping[str, NumericRange]
    calibration: Mapping[str, Any]
    joint_order: tuple[str, ...] = G1_JOINT_ORDER
    schema_version: int = FEASIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != FEASIBILITY_SCHEMA_VERSION
        ):
            raise FeasibilityConfigError(
                f"unsupported feasibility schema_version={self.schema_version!r}; expected {FEASIBILITY_SCHEMA_VERSION}"
            )
        joint_order = validate_joint_order(self.joint_order)
        parsed_ranges = {
            name: NumericRange.from_value(value, path=f"ranges.{name}") for name, value in self.ranges.items()
        }
        _expect_exact_keys(parsed_ranges, set(REQUIRED_FEASIBILITY_RANGES), "ranges")
        for name in _NONNEGATIVE_RANGE_NAMES:
            if parsed_ranges[name].minimum < 0.0:
                raise FeasibilityConfigError(f"ranges.{name}.min must be non-negative")
        # The limiter is undefined at zero and all force/drive magnitudes must
        # be strictly usable, not merely non-negative placeholders.
        for name in (
            "terrain.friction",
            "wheel.left_damping",
            "wheel.right_damping",
        ):
            if parsed_ranges[name].minimum <= 0.0:
                raise FeasibilityConfigError(f"ranges.{name}.min must be positive")
        calibration = _validate_calibration(self.calibration)
        for range_name, calibration_name in _NOMINAL_CALIBRATION_BY_RANGE.items():
            nominal = float(calibration[calibration_name])
            interval = parsed_ranges[range_name]
            if not interval.minimum <= nominal <= interval.maximum:
                raise FeasibilityConfigError(
                    f"calibration.{calibration_name}={nominal} lies outside ranges.{range_name}"
                )
        object.__setattr__(self, "joint_order", joint_order)
        object.__setattr__(self, "ranges", MappingProxyType(parsed_ranges))
        object.__setattr__(self, "calibration", calibration)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> FeasibilityEnvelope:
        data = _expect_mapping(mapping, "feasibility envelope")
        _expect_exact_keys(
            data,
            {"schema_version", "joint_order", "ranges", "calibration"},
            "feasibility envelope",
        )
        ranges = _flatten_interval_mapping(_expect_mapping(data["ranges"], "ranges"))
        calibration = _expect_mapping(data["calibration"], "calibration")
        return cls(
            schema_version=data["schema_version"],
            joint_order=data["joint_order"],
            ranges=ranges,
            calibration=calibration,
        )


def _load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise FeasibilityConfigError(f"duplicate YAML key {key!r} in {path}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise FeasibilityConfigError(f"invalid YAML in {file_path}: {exc}") from exc
    if value is None:
        raise FeasibilityConfigError(f"configuration file is empty: {file_path}")
    return _expect_mapping(value, str(file_path))


def load_feasibility_envelope(path: str | Path) -> FeasibilityEnvelope:
    """Load and fully validate a feasibility envelope YAML file."""

    file_path = Path(path)
    return FeasibilityEnvelope.from_mapping(_load_yaml_mapping(file_path))
