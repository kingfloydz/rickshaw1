"""Task-specific models built on the RSL-RL 5.4.0 model API."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict
from torch import nn

from g1_rickshaw_lab.policy_schema import (
    ACTOR_OBSERVATION_DIM,
    DEFAULT_CONTEXT_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
    validate_context_dim,
    validate_history_length,
)

from .context_encoder import ContextEncoder
from .teacher_model import TeacherEncoder

_ACTOR_HIDDEN_DIMS = (512, 256, 128)


def _ordered_state_dict(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    migrated = OrderedDict()
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        migrated._metadata = metadata.copy()  # type: ignore[attr-defined]
    return migrated


class RslRickshawActorModel(MLPModel):
    """Teacher or student context encoder with the standard RSL-RL MLP head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = _ACTOR_HIDDEN_DIMS,
        activation: str = "elu",
        obs_normalization: bool = False,
        cnn_cfg: dict[str, Any] | None = None,
        distribution_cfg: dict | None = None,
        rnn_type: str | None = None,
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        latent_dim: int = DEFAULT_CONTEXT_DIM,
        history_length: int = HISTORY_LENGTH,
    ) -> None:
        del cnn_cfg, rnn_type, rnn_hidden_dim, rnn_num_layers
        self.latent_dim = validate_context_dim(latent_dim)
        self.history_length = validate_history_length(history_length)
        self.stage = ""
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        if self.stage == "teacher":
            self.encoder = TeacherEncoder(self.latent_dim, self.history_length)
            self.dynamic_obs_normalizer = (
                EmpiricalNormalization(TEACHER_DYNAMIC_DIM) if obs_normalization else nn.Identity()
            )
            self.static_obs_normalizer = (
                EmpiricalNormalization(TEACHER_STATIC_DIM) if obs_normalization else nn.Identity()
            )
        else:
            self.encoder = ContextEncoder(self.latent_dim, self.history_length)
            self.dynamic_obs_normalizer = nn.Identity()
            self.static_obs_normalizer = nn.Identity()

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        active_groups = list(obs_groups[obs_set])
        self.stage = "teacher" if "teacher_static" in active_groups else "student"
        return active_groups, ACTOR_OBSERVATION_DIM

    def _get_latent_dim(self) -> int:
        return ACTOR_OBSERVATION_DIM + self.latent_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        del masks, hidden_state
        policy_sequence = self.obs_normalizer(obs["policy_sequence"])
        current = policy_sequence[:, -1]
        history = policy_sequence[:, :-1]
        if self.stage == "teacher":
            dynamic_sequence = self.dynamic_obs_normalizer(obs["teacher_dynamic_sequence"])
            context = self.encoder(
                history,
                dynamic_sequence[:, :-1],
                self.static_obs_normalizer(obs["teacher_static"]),
            )
        else:
            context = self.encoder(history)
        return torch.cat((current, context), dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        self.obs_normalizer.update(obs["policy_sequence"][:, -1])
        if self.stage == "teacher":
            self.dynamic_obs_normalizer.update(obs["teacher_dynamic_sequence"][:, -1])
            self.static_obs_normalizer.update(obs["teacher_static"])

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Load both native RSL-RL keys and checkpoints from the previous adapter."""

        migrated = _ordered_state_dict(state_dict)
        for key, value in state_dict.items():
            if key.startswith("policy.network."):
                key = "mlp." + key.removeprefix("policy.network.")
            elif key == "policy.std_param":
                key = "distribution.std_param"
            elif key.startswith("policy_obs_normalizer."):
                key = "obs_normalizer." + key.removeprefix("policy_obs_normalizer.")
            migrated[key] = value
        return super().load_state_dict(migrated, strict=strict, assign=assign)

    def as_jit(self) -> nn.Module:
        if self.stage != "student":
            raise RuntimeError("only the student actor is deployable")
        return _StudentExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        if self.stage != "student":
            raise RuntimeError("only the student actor is deployable")
        return _StudentOnnxExport(self, verbose)


class _StudentExport(nn.Module):
    def __init__(self, model: RslRickshawActorModel) -> None:
        super().__init__()
        self.context_encoder = _DeploymentContextEncoder(model.encoder)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.policy = copy.deepcopy(model.mlp)

    def forward(self, current: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        current = self.obs_normalizer(current)
        history = self.obs_normalizer(history)
        context = self.context_encoder(history)
        return self.policy(torch.cat((current, context), dim=-1))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _StudentOnnxExport(_StudentExport):
    def __init__(self, model: RslRickshawActorModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self):
        return (
            torch.zeros(1, ACTOR_OBSERVATION_DIM),
            torch.zeros(1, self.context_encoder.history_length, ACTOR_OBSERVATION_DIM),
        )

    @property
    def input_names(self) -> list[str]:
        return ["current", "history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _DeploymentContextEncoder(nn.Module):
    """Scriptable copy of the student TCN."""

    def __init__(self, encoder: ContextEncoder) -> None:
        super().__init__()
        self.history_length = encoder.history_length
        self.input = copy.deepcopy(encoder.input)
        self.blocks = copy.deepcopy(encoder.blocks)
        self.context = copy.deepcopy(encoder.context)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.input(history.transpose(1, 2)))[:, :, -1]
        return self.context(features)
