"""Gaussian actor and raw-privilege critic for G1 rickshaw control."""

from __future__ import annotations

import torch
from rsl_rl.modules import MLP, EmpiricalNormalization
from torch import nn
from torch.distributions import Independent, Normal

from g1_rickshaw_lab.policy_schema import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_DIM,
    DEFAULT_CONTEXT_DIM,
    HISTORY_LENGTH,
    validate_context_dim,
)

from .context_encoder import ContextEncoder

CURRENT_OBSERVATION_DIM = ACTOR_OBSERVATION_DIM
CRITIC_PRIVILEGE_DIM = CRITIC_PRIVILEGED_DIM
ACTOR_HIDDEN_DIMS = (512, 256, 128)
CRITIC_HIDDEN_DIMS = (512, 256, 128)


def _matrix(tensor: torch.Tensor, width: int, name: str) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}]")


class PolicyObservationNormalizer(EmpiricalNormalization):
    """RSL-RL empirical normalizer specialized to policy observations."""

    def __init__(self) -> None:
        super().__init__(ACTOR_OBSERVATION_DIM)


class GaussianActor(nn.Module):
    """Map the current observation and selected context to 29 actions."""

    current_dim = CURRENT_OBSERVATION_DIM
    action_dim = ACTION_DIM

    def __init__(self, latent_dim: int = DEFAULT_CONTEXT_DIM) -> None:
        super().__init__()
        self.latent_dim = validate_context_dim(latent_dim)
        self.network = MLP(
            CURRENT_OBSERVATION_DIM + self.latent_dim,
            ACTION_DIM,
            ACTOR_HIDDEN_DIMS,
            "elu",
        )
        self.std_param = nn.Parameter(torch.ones(ACTION_DIM))

    def distribution(
        self, current: torch.Tensor, context: torch.Tensor
    ) -> Independent:
        _matrix(current, CURRENT_OBSERVATION_DIM, "current")
        _matrix(context, self.latent_dim, "context")
        if current.shape[0] != context.shape[0]:
            raise ValueError("current and context batch dimensions differ")
        mean = self.network(torch.cat((current, context), dim=-1))
        std = self.std_param.clamp(1.0e-6, 1.0e6).to(dtype=mean.dtype).expand_as(mean)
        return Independent(Normal(mean, std, validate_args=False), 1)

    def forward(self, current: torch.Tensor, context: torch.Tensor) -> Independent:
        return self.distribution(current, context)

    def act(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        distribution = self.distribution(current, context)
        return distribution.mean if deterministic else distribution.sample()

    @property
    def std(self) -> torch.Tensor:
        return self.std_param.clamp(1.0e-6, 1.0e6)


class PrivilegedCritic(nn.Module):
    """Independent value network using current observation and raw privilege."""

    current_dim = CURRENT_OBSERVATION_DIM
    privileged_dim = CRITIC_PRIVILEGE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.network = MLP(
            CURRENT_OBSERVATION_DIM + CRITIC_PRIVILEGE_DIM,
            1,
            CRITIC_HIDDEN_DIMS,
            "elu",
        )

    def forward(
        self, current: torch.Tensor, privileged: torch.Tensor
    ) -> torch.Tensor:
        _matrix(current, CURRENT_OBSERVATION_DIM, "current")
        _matrix(privileged, self.privileged_dim, "privileged")
        if current.shape[0] != privileged.shape[0]:
            raise ValueError("current and privileged batch dimensions differ")
        return self.network(torch.cat((current, privileged), dim=-1))


class G1RickshawStudentActor(nn.Module):
    """Deployable student composed only of observation TCN and actor."""

    def __init__(
        self,
        latent_dim: int = DEFAULT_CONTEXT_DIM,
        history_length: int = HISTORY_LENGTH,
    ) -> None:
        super().__init__()
        self.obs_normalizer = PolicyObservationNormalizer()
        self.context_encoder = ContextEncoder(latent_dim, history_length)
        self.actor = GaussianActor(latent_dim)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(self.obs_normalizer(history))

    def forward_with_context(
        self, current: torch.Tensor, history: torch.Tensor
    ) -> tuple[Independent, torch.Tensor]:
        context = self.encode(history)
        return self.actor(self.obs_normalizer(current), context), context

    def forward(
        self, current: torch.Tensor, history: torch.Tensor
    ) -> Independent:
        return self.actor(self.obs_normalizer(current), self.encode(history))

    def act(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.actor.act(
            self.obs_normalizer(current),
            self.encode(history),
            deterministic=deterministic,
        )


__all__ = [
    "ACTION_DIM",
    "ACTOR_HIDDEN_DIMS",
    "CRITIC_PRIVILEGE_DIM",
    "CRITIC_HIDDEN_DIMS",
    "CURRENT_OBSERVATION_DIM",
    "PolicyObservationNormalizer",
    "G1RickshawStudentActor",
    "GaussianActor",
    "PrivilegedCritic",
]
