"""Rickshaw velocity command and flat-ground velocity projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply


def rickshaw_velocity(asset: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normal_w = torch.zeros_like(asset.data.root_link_lin_vel_w)
    normal_w[:, 2] = 1.0
    axle_b = torch.zeros_like(normal_w)
    axle_b[:, 1] = 1.0
    axle_w = quat_apply(asset.data.root_link_quat_w, axle_b)
    forward_w = torch.nn.functional.normalize(
        torch.cross(axle_w, normal_w, dim=-1), dim=-1
    )
    lateral_w = torch.cross(normal_w, forward_w, dim=-1)
    linear = asset.data.root_link_lin_vel_w
    return (
        torch.sum(linear * forward_w, dim=-1),
        torch.sum(linear * lateral_w, dim=-1),
        torch.sum(asset.data.root_link_ang_vel_w * normal_w, dim=-1),
    )


@dataclass(kw_only=True)
class RickshawVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return RickshawVelocityCommand(self, env)


class RickshawVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg: RickshawVelocityCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self.metrics = {
            "error_lin_vel_x": torch.zeros(self.num_envs, device=self.device),
            "error_ang_vel_z": torch.zeros(self.num_envs, device=self.device),
        }

    def _update_metrics(self) -> None:
        lin_vel_x, _, ang_vel_z = rickshaw_velocity(self.robot)
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_lin_vel_x"] += (
            torch.abs(self.vel_command_b[:, 0] - lin_vel_x) / max_command_step
        )
        self.metrics["error_ang_vel_z"] += (
            torch.abs(self.vel_command_b[:, 2] - ang_vel_z) / max_command_step
        )


__all__ = [
    "RickshawVelocityCommand",
    "RickshawVelocityCommandCfg",
    "rickshaw_velocity",
]
