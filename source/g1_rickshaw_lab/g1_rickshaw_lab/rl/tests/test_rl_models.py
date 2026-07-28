"""Pure-PyTorch tests for the teacher, student, actor, and critic."""

from __future__ import annotations

import unittest

import torch
from rsl_rl.modules import MLP, EmpiricalNormalization
from torch import nn

from g1_rickshaw_lab.policy_schema import ACTOR_OBSERVATION_DIM
from g1_rickshaw_lab.rl import (
    CRITIC_PRIVILEGE_DIM,
    DYNAMIC_PRIVILEGE_DIM,
    STATIC_PRIVILEGE_DIM,
    ContextEncoder,
    G1RickshawStudentActor,
    G1RickshawTeacherActor,
    GaussianActor,
    PrivilegedCritic,
    temporal_receptive_field,
)
from g1_rickshaw_lab.rl.context_encoder import FEATURE_DIM


class TestRickshawRLModels(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_context_schema_and_receptive_field(self) -> None:
        encoder = ContextEncoder()
        self.assertEqual(encoder.receptive_field, temporal_receptive_field())
        self.assertEqual(encoder.receptive_field, 61)
        self.assertEqual(FEATURE_DIM, 48)
        self.assertEqual(encoder.input.out_channels, FEATURE_DIM)
        self.assertEqual(encoder(torch.zeros(2, 61, ACTOR_OBSERVATION_DIM)).shape, (2, 16))

        history = torch.full((1, 61, ACTOR_OBSERVATION_DIM), 0.01, requires_grad=True)
        encoder(history).sum().backward()
        self.assertGreater(history.grad[:, 0].abs().sum().item(), 0.0)

        probe = torch.full((1, 62, ACTOR_OBSERVATION_DIM), 0.01, requires_grad=True)
        feature = encoder.blocks(encoder.input(probe.transpose(1, 2)))[:, :, -1]
        encoder.context(feature).sum().backward()
        self.assertEqual(probe.grad[:, 0].count_nonzero().item(), 0)
        self.assertGreater(probe.grad[:, 1].abs().sum().item(), 0.0)

    def test_actor_and_critic_architectures(self) -> None:
        actor = GaussianActor()
        self.assertIsInstance(actor.network, MLP)
        actor_shapes = [
            (layer.in_features, layer.out_features) for layer in actor.network if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(
            actor_shapes,
            [(ACTOR_OBSERVATION_DIM + 16, 512), (512, 256), (256, 128), (128, 29)],
        )
        torch.testing.assert_close(actor.std, torch.ones(29))

        actor.std_param.data.fill_(0.25)
        distribution = actor.distribution(
            torch.zeros(2, ACTOR_OBSERVATION_DIM), torch.zeros(2, 16)
        )
        torch.testing.assert_close(
            distribution.base_dist.scale,
            torch.full((2, 29), 0.25),
        )

        critic = PrivilegedCritic()
        critic_shapes = [
            (layer.in_features, layer.out_features) for layer in critic.network if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(
            critic_shapes,
            [
                (ACTOR_OBSERVATION_DIM + CRITIC_PRIVILEGE_DIM, 512),
                (512, 256),
                (256, 128),
                (128, 1),
            ],
        )
        self.assertEqual(
            critic(
                torch.randn(3, ACTOR_OBSERVATION_DIM),
                torch.randn(3, CRITIC_PRIVILEGE_DIM),
            ).shape,
            (3, 1),
        )

    def test_teacher_and_student_share_the_actor_latent_abi(self) -> None:
        batch = 3
        current = torch.randn(batch, ACTOR_OBSERVATION_DIM)
        history = torch.randn(batch, 61, ACTOR_OBSERVATION_DIM)
        dynamic = torch.randn(batch, 61, DYNAMIC_PRIVILEGE_DIM)
        static = torch.randn(batch, STATIC_PRIVILEGE_DIM)

        teacher = G1RickshawTeacherActor()
        teacher_distribution, z_star = teacher.forward_with_context(current, history, dynamic, static)
        context_shapes = [
            (layer.in_features, layer.out_features)
            for layer in teacher.encoder.context
            if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(context_shapes, [(64, 48), (48, 16)])
        student = G1RickshawStudentActor()
        self.assertIsInstance(student.obs_normalizer, EmpiricalNormalization)
        self.assertEqual(student.context_encoder.context.in_features, 48)
        student_distribution, z_hat = student.forward_with_context(current, history)

        self.assertEqual(teacher_distribution.mean.shape, (batch, 29))
        self.assertEqual(student_distribution.mean.shape, (batch, 29))
        self.assertEqual(z_star.shape, (batch, 16))
        self.assertEqual(z_hat.shape, (batch, 16))
        self.assertEqual(set(teacher.actor.state_dict()), set(student.actor.state_dict()))

        for latent_dim in (8, 16, 24, 32):
            with self.subTest(latent_dim=latent_dim):
                teacher = G1RickshawTeacherActor(latent_dim)
                student = G1RickshawStudentActor(latent_dim)
                teacher_distribution, z_star = teacher.forward_with_context(current, history, dynamic, static)
                student_distribution, z_hat = student.forward_with_context(current, history)
                self.assertEqual(z_star.shape, (batch, latent_dim))
                self.assertEqual(z_hat.shape, (batch, latent_dim))
                self.assertEqual(
                    teacher.actor.network[0].in_features,
                    ACTOR_OBSERVATION_DIM + latent_dim,
                )
                self.assertEqual(
                    student.actor.network[0].in_features,
                    ACTOR_OBSERVATION_DIM + latent_dim,
                )
                self.assertEqual(
                    teacher_distribution.mean.shape,
                    student_distribution.mean.shape,
                )

if __name__ == "__main__":
    unittest.main()
