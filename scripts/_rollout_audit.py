"""Pure CPU structural contracts for S1 teacher rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from g1_rickshaw_lab.training_contract import (
    GUIDE_TRAINING_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION,
)

ACTION_DIM = 29
DEFAULT_NUM_ENVS = GUIDE_TRAINING_NUM_ENVS
AUDIT_TENSOR_NAMES = (
    "curriculum_stage",
    "collection_segment",
    "environment_id",
    "episode_id",
)
INTEGER_AUDIT_TENSORS = frozenset(AUDIT_TENSOR_NAMES)


def _column(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"rollout audit tensor {name} must have shape [N,1]")
    return value


def normalize_audit_tensors(raw: Mapping[str, Any], *, batch_size: int) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in AUDIT_TENSOR_NAMES:
        if name not in raw:
            raise ValueError(f"rollout shard is missing audit tensor {name!r}")
        tensor = _column(torch.as_tensor(raw[name]).detach().cpu(), name)
        if tensor.is_floating_point() and not torch.all(tensor == torch.round(tensor)):
            raise ValueError(f"rollout audit tensor {name} contains non-integer values")
        if tensor.shape[0] != batch_size:
            raise ValueError(f"rollout audit tensor {name} has the wrong sample count")
        result[name] = tensor.long()
    return result


def _validate_episode_binding(tensors: Mapping[str, torch.Tensor]) -> int:
    episodes = tensors["episode_id"].reshape(-1)
    if episodes.numel() == 0 or torch.any(episodes < 0):
        raise ValueError("rollout episode_id must be non-negative")
    order = torch.argsort(episodes, stable=True)
    same = episodes[order][1:] == episodes[order][:-1]
    for name in AUDIT_TENSOR_NAMES:
        if name == "episode_id":
            continue
        values = tensors[name][order].reshape(episodes.numel(), -1)
        if torch.any(same & ~(values[1:] == values[:-1]).all(dim=1)):
            raise ValueError(f"one episode changes {name} without a reset")
    return int(torch.unique(episodes).numel())


def summarize_segment_samples(
    tensors: Mapping[str, torch.Tensor],
    *,
    segment_index: int,
    num_envs: int,
    samples_per_environment: int,
) -> dict[str, Any]:
    """Validate and summarize the nineteen-slope TRAINING segment."""

    if segment_index != 0:
        raise ValueError("collection segment must be 0")
    expected_samples = num_envs * samples_per_environment
    if tensors["environment_id"].shape[0] != expected_samples:
        raise ValueError("TRAINING segment has the wrong sample count")
    if not torch.all(tensors["collection_segment"] == 0):
        raise ValueError("rollout contains a non-TRAINING collection segment")
    if not torch.all(tensors["curriculum_stage"] == 1):
        raise ValueError("rollout contains a non-TRAINING curriculum stage")
    environment_ids = tensors["environment_id"].reshape(-1)
    if torch.any(environment_ids < 0) or torch.any(environment_ids >= num_envs):
        raise ValueError("rollout environment_id is out of range")
    if not torch.all(torch.bincount(environment_ids, minlength=num_envs) == samples_per_environment):
        raise ValueError("every rollout environment must contribute the same quota")
    return {
        "segment_index": 0,
        "samples": expected_samples,
        "episodes": _validate_episode_binding(tensors),
        "stage_distribution": {"TRAINING": expected_samples},
        "environment_stage_distribution": {"TRAINING": num_envs},
    }


def validate_rollout_sample_audit(
    manifest: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Recompute the rollout structure and coverage audit."""

    if manifest.get("schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"S1 rollout requires manifest schema {ROLLOUT_MANIFEST_SCHEMA_VERSION}")
    audit = manifest.get("sample_audit")
    if not isinstance(audit, Mapping) or audit.get("schema_version") != ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION:
        raise ValueError("rollout manifest is missing its sample audit")
    segments = manifest.get("stage_segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError("S1 rollout requires exactly one TRAINING segment")
    summary = summarize_segment_samples(
        tensors,
        segment_index=0,
        num_envs=int(manifest.get("num_envs", 0)),
        samples_per_environment=int(manifest.get("num_steps_per_stage", 0)),
    )
    segment = segments[0]
    if segment.get("global_stage") != "TRAINING" or segment.get("actual_sample_audit") != summary:
        raise ValueError("TRAINING segment manifest differs from shard samples")
    return {
        "manifest_schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "sample_audit_schema_version": ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION,
        "stages": {"TRAINING": summary},
    }


__all__ = [
    "ACTION_DIM",
    "AUDIT_TENSOR_NAMES",
    "DEFAULT_NUM_ENVS",
    "INTEGER_AUDIT_TENSORS",
    "ROLLOUT_MANIFEST_SCHEMA_VERSION",
    "ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION",
    "normalize_audit_tensors",
    "summarize_segment_samples",
    "validate_rollout_sample_audit",
]
