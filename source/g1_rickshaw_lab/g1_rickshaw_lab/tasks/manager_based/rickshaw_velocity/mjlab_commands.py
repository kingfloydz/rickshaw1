"""Rickshaw velocity command and flat-ground velocity projection."""

from __future__ import annotations

from collections.abc import Callable
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
        self._rickshaw_joystick_enabled = None
        self._rickshaw_joystick_sliders: dict[int, Any] = {}

    def create_gui(
        self,
        name: str,
        server: Any,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Create controls for the two command axes used by the rickshaw task."""

        del on_change, request_action
        from viser import Icon

        axes = (
            (0, "lin_vel_x", self.cfg.ranges.lin_vel_x),
            (2, "ang_vel_z", self.cfg.ranges.ang_vel_z),
        )
        sliders: dict[int, Any] = {}
        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)
            for command_index, label, limits in axes:
                max_value = max(abs(float(limits[0])), abs(float(limits[1])))
                max_input = server.gui.add_slider(
                    f"Max {label}",
                    initial_value=max_value,
                    step=0.1,
                    min=0.1,
                    max=10.0,
                )
                slider = server.gui.add_slider(
                    label,
                    min=-max_value,
                    max=max_value,
                    step=0.05,
                    initial_value=0.0,
                )

                @max_input.on_update
                def _update_range(_event, target=slider, maximum=max_input) -> None:
                    target.min = -maximum.value
                    target.max = maximum.value

                sliders[command_index] = slider

            zero_button = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_button.on_click
            def _zero(_event) -> None:
                for slider in sliders.values():
                    slider.value = 0.0

        self._rickshaw_joystick_enabled = enabled
        self._rickshaw_joystick_sliders = sliders
        self._rickshaw_joystick_get_env_idx = get_env_idx

    def compute(self, dt: float) -> None:
        super().compute(dt)
        enabled = self._rickshaw_joystick_enabled
        if enabled is not None and enabled.value:
            env_idx = self._rickshaw_joystick_get_env_idx()
            self.vel_command_b[env_idx, 1] = 0.0
            for command_index, slider in self._rickshaw_joystick_sliders.items():
                self.vel_command_b[env_idx, command_index] = slider.value

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
