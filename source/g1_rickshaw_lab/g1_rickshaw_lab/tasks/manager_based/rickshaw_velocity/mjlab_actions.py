"""Mjlab action term with a per-environment static-equilibrium reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from g1_rickshaw_lab.policy_schema import ACTION_SCALE


@dataclass(kw_only=True)
class StaticReferenceJointPositionActionCfg(BaseActionCfg):
    """Joint-position action centered on the reset solver's current ``q_ref``."""

    def build(self, env):
        return StaticReferenceJointPositionAction(self, env)


class StaticReferenceJointPositionAction(BaseAction):
    cfg: StaticReferenceJointPositionActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.q_ref = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._scale = torch.tensor(ACTION_SCALE, device=self.device).unsqueeze(0)
        self._processed_actions[:] = self.q_ref

    def set_reference(
        self,
        q_ref: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        self.q_ref[env_ids] = q_ref
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = q_ref

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.q_ref + actions * self._scale

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(
            self._processed_actions - encoder_bias,
            joint_ids=self._target_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = self.q_ref[ids]


__all__ = [
    "StaticReferenceJointPositionAction",
    "StaticReferenceJointPositionActionCfg",
]
