"""Acceptance tests for the fixed actor observation and causal history."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from g1_rickshaw_lab.policy_schema import (
    TEACHER_STATIC_DIM,
    TEACHER_TERRAIN_SLOPE_BOUNDS,
)
from g1_rickshaw_lab.rl import DYNAMIC_PRIVILEGE_DIM, STATIC_PRIVILEGE_DIM
from g1_rickshaw_lab.rl.actor_critic import G1RickshawStudentActor
from g1_rickshaw_lab.rl.teacher_model import G1RickshawTeacherActor
from g1_rickshaw_lab.rl.context_encoder import (
    DILATIONS,
    HISTORY_LENGTH as TCN_HISTORY_LENGTH,
    KERNEL_SIZE,
    ContextEncoder,
    temporal_receptive_field,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.observations import (
    ACTOR_OBSERVATION_DIM,
    BASE_ANGULAR_VELOCITY_SLICE,
    BASE_LINEAR_VELOCITY_SLICE,
    COMMAND_SLICE,
    HISTORY_LENGTH,
    JOINT_POSITION_SLICE,
    JOINT_VELOCITY_SLICE,
    PREVIOUS_ACTION_SLICE,
    PROJECTED_GRAVITY_SLICE,
    TEACHER_STATIC_FEATURE_NAMES,
    ObservationHistoryState,
    assemble_actor_observation,
    assemble_teacher_dynamic_privilege,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    _update_teacher_static_domain,
)


def test_actor_observation_has_the_fixed_scaled_order() -> None:
    dtype = torch.float64
    linear_velocity = torch.tensor([[1.2, -0.4, 0.3]], dtype=dtype)
    angular_velocity = torch.tensor([[4.0, -2.0, 1.0]], dtype=dtype)
    gravity = torch.tensor([[0.1, 0.2, -0.9]], dtype=dtype)
    command = torch.tensor([[0.7, -0.2]], dtype=dtype)
    q_ref = torch.linspace(-0.2, 0.2, 29, dtype=dtype).unsqueeze(0)
    position_error = torch.linspace(-0.1, 0.1, 29, dtype=dtype).unsqueeze(0)
    joint_position = q_ref + position_error
    joint_velocity = torch.linspace(-2.0, 2.0, 29, dtype=dtype).unsqueeze(0)
    previous_action = torch.linspace(-0.5, 0.5, 29, dtype=dtype).unsqueeze(0)

    observation = assemble_actor_observation(
        linear_velocity,
        angular_velocity,
        gravity,
        command,
        joint_position,
        q_ref,
        joint_velocity,
        previous_action,
    )

    assert observation.shape == (1, ACTOR_OBSERVATION_DIM)
    assert ACTOR_OBSERVATION_DIM == 98
    torch.testing.assert_close(
        observation[:, BASE_LINEAR_VELOCITY_SLICE], linear_velocity
    )
    torch.testing.assert_close(
        observation[:, BASE_ANGULAR_VELOCITY_SLICE], angular_velocity * 0.25
    )
    torch.testing.assert_close(observation[:, PROJECTED_GRAVITY_SLICE], gravity)
    torch.testing.assert_close(observation[:, COMMAND_SLICE], command)
    torch.testing.assert_close(observation[:, JOINT_POSITION_SLICE], position_error)
    torch.testing.assert_close(
        observation[:, JOINT_VELOCITY_SLICE], joint_velocity * 0.05
    )
    torch.testing.assert_close(observation[:, PREVIOUS_ACTION_SLICE], previous_action)


def test_teacher_dynamic_privilege_keeps_separate_hand_forces() -> None:
    dtype = torch.float64
    axes = torch.eye(3, dtype=dtype)
    result = assemble_teacher_dynamic_privilege(
        torch.tensor([[0.6, -0.1]], dtype=dtype),
        torch.tensor([0.2], dtype=dtype),
        torch.tensor([[10.0, 20.0]], dtype=dtype),
        torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=dtype),
        axes[0:1],
        axes[1:2],
        axes[2:3],
    )

    torch.testing.assert_close(
        result,
        torch.tensor(
            [[0.6, -0.1, 0.2, 10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            dtype=dtype,
        ),
    )


def test_teacher_static_and_critic_privilege_include_normalized_terrain_slope() -> None:
    dtype = torch.float64
    slopes = torch.tensor((-0.08, 0.01, 0.10), dtype=dtype)
    env = SimpleNamespace(
        num_envs=slopes.numel(),
        device=torch.device("cpu"),
        path_tangent_w=torch.stack(
            (torch.cos(slopes), torch.zeros_like(slopes), torch.sin(slopes)), dim=-1
        ),
        effective_torso_mass=torch.full((slopes.numel(),), 30.0, dtype=dtype),
        effective_cart_mass_com=torch.tensor((35.04, 0.1, 0.0, 0.3), dtype=dtype)
        .expand(slopes.numel(), -1)
        .clone(),
        _default_robot_masses_cpu=torch.tensor(((30.0,),), dtype=dtype),
        torso_body_id=0,
    )
    cfg = SimpleNamespace(
        ranges={
            "torso.mass_delta": (-1.0, 1.0),
            "payload.mass": (0.0, 10.0),
            "payload.com.x": (-0.1, 0.1),
            "payload.com.y": (-0.1, 0.1),
            "payload.com.z": (-0.1, 0.1),
            "rolling_resistance.c_rr": (0.01, 0.1),
            "terrain.friction": (0.5, 1.5),
            "wheel.left_damping": (0.01, 0.1),
            "wheel.right_damping": (0.01, 0.1),
        }
    )
    sampled = {
        "rolling_resistance.c_rr": torch.full_like(slopes, 0.05),
        "terrain.friction": torch.full_like(slopes, 0.8),
        "wheel.left_damping": torch.full_like(slopes, 0.02),
        "wheel.right_damping": torch.full_like(slopes, 0.02),
    }

    _update_teacher_static_domain(env, cfg, sampled)

    assert TEACHER_STATIC_FEATURE_NAMES[-1] == "terrain.slope"
    assert env.teacher_static_domain_raw.shape == (slopes.numel(), TEACHER_STATIC_DIM)
    torch.testing.assert_close(env.teacher_static_domain_raw[:, -1], slopes)
    assert (
        tuple(bound[-1].item() for bound in env.teacher_static_domain_bounds)
        == TEACHER_TERRAIN_SLOPE_BOUNDS
    )
    torch.testing.assert_close(
        env.normalized_teacher_static_domain[:, -1],
        torch.tensor((-1.0, 0.0, 1.0), dtype=dtype),
    )


def test_history_excludes_current_observation() -> None:
    state = ObservationHistoryState.zeros(1, dtype=torch.float64)
    first = torch.full((1, ACTOR_OBSERVATION_DIM), 10.0, dtype=torch.float64)
    second = torch.full((1, ACTOR_OBSERVATION_DIM), 20.0, dtype=torch.float64)
    third = torch.full((1, ACTOR_OBSERVATION_DIM), 30.0, dtype=torch.float64)

    state.advance(first)
    assert state.history.shape == (1, HISTORY_LENGTH, ACTOR_OBSERVATION_DIM)
    assert (HISTORY_LENGTH, ACTOR_OBSERVATION_DIM) == (61, 98)
    torch.testing.assert_close(state.history, first[:, None, :].expand(-1, 61, -1))
    torch.testing.assert_close(state.current, first)

    state.advance(second)
    torch.testing.assert_close(state.history[:, -1], first)
    torch.testing.assert_close(state.current, second)
    assert not torch.any(state.history == 20.0)

    state.advance(third)
    torch.testing.assert_close(state.history[:, -2], first)
    torch.testing.assert_close(state.history[:, -1], second)
    torch.testing.assert_close(state.current, third)
    assert not torch.any(state.history == 30.0)

    frozen_history = state.history.clone()
    frozen_current = state.current.clone()
    state.advance(torch.full_like(third, 40.0), valid_mask=torch.tensor([False]))
    torch.testing.assert_close(state.history, frozen_history)
    torch.testing.assert_close(state.current, frozen_current)


def test_history_state_can_track_current_without_allocating_temporal_storage() -> None:
    state = ObservationHistoryState.zeros(2, history_enabled=False)
    observation = torch.randn(2, ACTOR_OBSERVATION_DIM)

    state.advance(observation)

    assert state.history is None
    torch.testing.assert_close(state.current, observation)
    assert torch.all(state.initialized)


def test_tcn_schema_receptive_field_and_single_history_path() -> None:
    assert TCN_HISTORY_LENGTH == HISTORY_LENGTH == 61
    assert KERNEL_SIZE == 5
    assert DILATIONS == (1, 2, 4, 8)
    assert temporal_receptive_field() == 61

    encoder = ContextEncoder().eval()
    blocks = list(encoder.blocks)
    assert tuple(block.conv.dilation[0] for block in blocks) == DILATIONS
    for block in blocks:
        dilated_convolutions = [
            module
            for module in block.modules()
            if isinstance(module, nn.Conv1d) and module.kernel_size == (KERNEL_SIZE,)
        ]
        assert len(dilated_convolutions) == 1
        assert dilated_convolutions[0].stride == (1,)

    assert encoder(torch.zeros(2, 61, ACTOR_OBSERVATION_DIM)).shape == (2, 16)
    expected_shape = rf"\[N, 61, {ACTOR_OBSERVATION_DIM}\]"
    with pytest.raises(ValueError, match=expected_shape):
        encoder(torch.zeros(2, 60, ACTOR_OBSERVATION_DIM))
    with pytest.raises(ValueError, match=expected_shape):
        encoder(torch.zeros(2, 61, ACTOR_OBSERVATION_DIM - 1))

    student = G1RickshawStudentActor()
    history_encoders = [
        module for module in student.modules() if isinstance(module, ContextEncoder)
    ]
    recurrent_modules = [
        module
        for module in student.modules()
        if isinstance(module, (nn.RNNBase, nn.RNNCellBase))
    ]
    assert len(history_encoders) == 1
    assert recurrent_modules == []


def test_fixed_teacher_and_student_context_interfaces() -> None:
    teacher = G1RickshawTeacherActor().eval()
    student = G1RickshawStudentActor().eval()
    current = torch.zeros(2, ACTOR_OBSERVATION_DIM)
    observation_history = torch.zeros(2, 61, ACTOR_OBSERVATION_DIM)
    dynamic_history = torch.zeros(2, 61, DYNAMIC_PRIVILEGE_DIM)
    static_privilege = torch.zeros(2, STATIC_PRIVILEGE_DIM)
    with torch.no_grad():
        teacher_distribution, teacher_context = teacher.forward_with_context(
            current,
            observation_history,
            dynamic_history,
            static_privilege,
        )
        student_distribution, student_context = student.forward_with_context(
            current, observation_history
        )
    assert teacher.encoder.latent_dim == 16
    assert student.context_encoder.latent_dim == 16
    teacher_context_layers = list(teacher.encoder.context)
    assert isinstance(teacher_context_layers[0], nn.Linear)
    assert teacher_context_layers[0].in_features == 96
    assert teacher_context_layers[0].out_features == 64
    assert isinstance(teacher_context_layers[1], nn.ELU)
    assert isinstance(teacher_context_layers[2], nn.Linear)
    assert teacher_context_layers[2].in_features == 64
    assert teacher_context_layers[2].out_features == 16
    assert teacher_context.shape == (2, 16)
    assert student_context.shape == (2, 16)
    assert teacher_distribution.mean.shape == (2, 29)
    assert student_distribution.mean.shape == (2, 29)
    assert teacher.actor.network[0].in_features == ACTOR_OBSERVATION_DIM + 16
    assert student.actor.network[0].in_features == ACTOR_OBSERVATION_DIM + 16
    assert not hasattr(teacher, "context_projection")
    assert not hasattr(student, "context_projection")
    assert not any("aux" in name for name, _ in teacher.named_modules())
    assert not any("aux" in name for name, _ in student.named_modules())


@pytest.mark.parametrize("latent_dim", (8, 16, 24, 32))
def test_teacher_and_student_use_the_selected_latent_width(latent_dim: int) -> None:
    teacher = G1RickshawTeacherActor(latent_dim).eval()
    student = G1RickshawStudentActor(latent_dim).eval()
    current = torch.zeros(2, ACTOR_OBSERVATION_DIM)
    history = torch.zeros(2, 61, ACTOR_OBSERVATION_DIM)
    with torch.no_grad():
        teacher_distribution, teacher_context = teacher.forward_with_context(
            current,
            history,
            torch.zeros(2, 61, DYNAMIC_PRIVILEGE_DIM),
            torch.zeros(2, STATIC_PRIVILEGE_DIM),
        )
        student_distribution, student_context = student.forward_with_context(
            current, history
        )

    assert teacher_context.shape == student_context.shape == (2, latent_dim)
    assert teacher.actor.network[0].in_features == ACTOR_OBSERVATION_DIM + latent_dim
    assert student.actor.network[0].in_features == ACTOR_OBSERVATION_DIM + latent_dim
    assert set(teacher.actor.state_dict()) == set(student.actor.state_dict())
    assert teacher_distribution.mean.shape == student_distribution.mean.shape == (2, 29)


def test_tcn_oldest_frame_is_used_but_outside_and_future_frames_are_causal() -> None:
    encoder = ContextEncoder().eval()
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.fill_(0.01)

    # At output index 61 the exact 61-frame receptive field is input 1..61.
    probe = torch.full((1, 62, ACTOR_OBSERVATION_DIM), 0.01)
    outside_perturbed = probe.clone()
    outside_perturbed[:, 0] += 10.0
    with torch.no_grad():
        base_feature = encoder.blocks(encoder.input(probe.transpose(1, 2)))[:, :, -1]
        perturbed_feature = encoder.blocks(
            encoder.input(outside_perturbed.transpose(1, 2))
        )[:, :, -1]
        base_context = encoder.context(base_feature)
        perturbed_context = encoder.context(perturbed_feature)
    torch.testing.assert_close(base_context, perturbed_context, rtol=0.0, atol=0.0)

    oldest_perturbed = probe.clone()
    oldest_perturbed[:, 1] += 0.1
    with torch.no_grad():
        oldest_feature = encoder.blocks(
            encoder.input(oldest_perturbed.transpose(1, 2))
        )[:, :, -1]
        oldest_context = encoder.context(oldest_feature)
    assert not torch.allclose(base_context, oldest_context)

    prefix = torch.randn(1, 61, ACTOR_OBSERVATION_DIM)
    future = torch.randn(1, 7, ACTOR_OBSERVATION_DIM)
    with torch.no_grad():
        prefix_outputs = encoder.blocks(encoder.input(prefix.transpose(1, 2)))
        extended_outputs = encoder.blocks(
            encoder.input(torch.cat((prefix, future), dim=1).transpose(1, 2))
        )
    torch.testing.assert_close(
        prefix_outputs, extended_outputs[:, :, :61], rtol=0.0, atol=0.0
    )
