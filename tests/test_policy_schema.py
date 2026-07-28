"""Cross-layer checks for the single policy ABI."""

from __future__ import annotations

import pytest

from g1_rickshaw_lab import policy_schema
from g1_rickshaw_lab.rl import context_encoder, teacher_model
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import observations


def test_policy_dimensions_are_shared_across_runtime_layers() -> None:
    assert context_encoder.OBSERVATION_DIM == policy_schema.ACTOR_OBSERVATION_DIM
    assert context_encoder.HISTORY_LENGTH == policy_schema.HISTORY_LENGTH
    assert policy_schema.DEFAULT_CONTEXT_DIM == 16
    assert policy_schema.SUPPORTED_CONTEXT_DIMS == (
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        24,
        32,
    )
    assert teacher_model.DYNAMIC_PRIVILEGE_DIM == policy_schema.TEACHER_DYNAMIC_DIM
    assert teacher_model.STATIC_PRIVILEGE_DIM == policy_schema.TEACHER_STATIC_DIM
    assert observations.ACTOR_OBSERVATION_DIM == policy_schema.ACTOR_OBSERVATION_DIM
    assert observations.HISTORY_LENGTH == policy_schema.HISTORY_LENGTH
    assert observations.TEACHER_DYNAMIC_DIM == policy_schema.TEACHER_DYNAMIC_DIM
    assert observations.TEACHER_STATIC_DIM == policy_schema.TEACHER_STATIC_DIM
    assert policy_schema.ACTOR_OBSERVATION_DIM == 98
    assert policy_schema.TEACHER_DYNAMIC_DIM == 11
    assert policy_schema.TEACHER_STATIC_DIM == 10
    assert policy_schema.CRITIC_PRIVILEGED_DIM == 35
    assert policy_schema.TEACHER_TERRAIN_SLOPE_BOUNDS == (-0.08, 0.10)
    assert observations.TEACHER_STATIC_FEATURE_NAMES[-1] == "terrain.slope"
    assert observations.TEACHER_DYNAMIC_FEATURE_NAMES == (
        "rickshaw.velocity.lin_vel_x",
        "rickshaw.velocity.ang_vel_z",
        "cart.pitch",
        "wheel.left_normal_force",
        "wheel.right_normal_force",
        "hand.left.force.s",
        "hand.left.force.l",
        "hand.left.force.n",
        "hand.right.force.s",
        "hand.right.force.l",
        "hand.right.force.n",
    )

    for latent_dim in policy_schema.SUPPORTED_CONTEXT_DIMS:
        assert policy_schema.validate_context_dim(latent_dim) == latent_dim
    with pytest.raises(ValueError, match="context dimension"):
        policy_schema.validate_context_dim(8.0)
